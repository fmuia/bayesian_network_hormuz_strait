"""Whole-network elicitation runner.

Recruits ``n_agents`` agents distributed across the framework's selected models,
scores them on the seeds once, then for each in-scope node: lets the LLM pick
node-appropriate roles, elicits the node's CPT from each agent (one call per
agent per node), and pools the columns with the Cooke weights — estimating a
per-node kappa from the panel's spread (correlation-discounted) and capping it
by the panel's calibration.

Effort controls bound cost/time: a node cap, cheap-nodes-first ordering,
bounded concurrency, per-call timeouts, and partial results (a node that errors
or times out is skipped and keeps its bootstrap CPT).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np

from ...network_spec import NetworkSpec
from ..engine.aggregation import cooke_pool
from ..engine.calibration import DEFAULT_QUANTILES, ExpertScore, classical_model_weights
from ..engine.kappa import KappaLadder, kappa_from_panel_spread
from ..agents.decorrelation import mean_pairwise_correlation
from .clients import ElicitationClient, make_client
from .framework import ElicitationFramework, ModelSpec

ProgressFn = Callable[[str, float], None]
Config = tuple[str, ...]

# Claiming a "tight" kappa needs at least this much panel calibration. Below it,
# strong agreement is reported as "normal", not "tight". Calibration never on its
# own forces "uncertain" — that comes from genuine disagreement (spread).
TIGHT_CALIBRATION_FLOOR = 0.05


@dataclass
class Agent:
    name: str
    model: str
    provider: str
    client: ElicitationClient


@dataclass
class NodeElicitation:
    node: str
    roles: list[str]
    rationales: dict[str, str]
    mean_columns: dict[Config, list[float]]
    per_expert: dict[str, dict[Config, list[float]]]
    kappa: float
    kappa_level: str
    error: str | None = None


@dataclass
class ElicitationRun:
    run_id: str
    created_at: str
    framework_name: str
    n_agents: int
    agent_names: list[str]
    models: list[str]
    expert_scores: list[dict]
    elicited_nodes: list[str]
    skipped_nodes: list[str]
    correlation_note: str
    spec: NetworkSpec
    nodes: dict[str, NodeElicitation] = field(default_factory=dict)
    seeds: list[dict] = field(default_factory=list)             # the calibration questions asked
    seed_answers: dict[str, dict[str, list[float]]] = field(default_factory=dict)  # agent -> {seed_id: quantiles}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _parent_configs(spec: NetworkSpec, node_name: str) -> list[Config]:
    node = spec.nodes[node_name]
    if not node.parents:
        return [()]
    return [tuple(c) for c in product(*[list(spec.states[p]) for p in node.parents])]


def _select_nodes(spec: NetworkSpec, selected: list[str]) -> list[str]:
    """The nodes to elicit. An explicit selection (empty = whole network); never
    silently drops a chosen node. Unselected nodes keep their bootstrap CPT."""
    if selected:
        return [n for n in selected if n in spec.nodes]
    return list(spec.nodes)


def build_agents(
    framework: ElicitationFramework,
    client_factory: Callable[[ModelSpec], ElicitationClient] | None = None,
) -> list[Agent]:
    """Distribute n_agents round-robin across the framework's models."""
    factory = client_factory or (
        lambda m: make_client(
            m.provider, m.model,
            framework.effort.time_limit_s, framework.effort.hard_timeout_s, framework.effort.reasoning_effort,
        )
    )
    models = framework.models or []
    if not models:
        raise ValueError("framework has no models selected")
    agents: list[Agent] = []
    for i in range(framework.n_agents):
        spec = models[i % len(models)]
        agents.append(
            Agent(
                name=f"agent-{i + 1} ({spec.model})",
                model=spec.model,
                provider=spec.provider,
                client=factory(spec),
            )
        )
    return agents


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def run_elicitation(
    base_spec: NetworkSpec,
    framework: ElicitationFramework,
    run_id: str,
    created_at: str,
    client_factory: Callable[[ModelSpec], ElicitationClient] | None = None,
    on_progress: ProgressFn | None = None,
    ladder: KappaLadder | None = None,
) -> ElicitationRun:
    progress = on_progress or (lambda *_: None)
    ladder = ladder or KappaLadder()
    levels = DEFAULT_QUANTILES
    agents = build_agents(framework, client_factory)
    seeds = framework.seeds[: framework.effort.n_seeds] if framework.effort.n_seeds else framework.seeds

    # 1. Score the panel on the seeds (role-neutral) once. With no seeds, fall
    #    back to equal weighting and no calibration evidence (kappa from agreement).
    seed_quantiles: dict[str, dict[str, list[float]]] = {}
    if seeds:
        progress("Scoring agents on seeds", 0.05)
        for agent in agents:
            answers, _ = agent.client.seed_quantiles(seeds, levels, role=None)
            seed_quantiles[agent.name] = answers
        experts_quantiles = [
            np.array([seed_quantiles[a.name][s.id] for s in seeds], dtype=float) for a in agents
        ]
        realizations = np.array([s.realization for s in seeds], dtype=float)
        scores = classical_model_weights(experts_quantiles, realizations, levels, alpha=0.0)
    else:
        scores = [
            ExpertScore(calibration=0.0, information=0.0, raw_weight=1.0, weight=1.0 / len(agents))
            for _ in agents
        ]
    weights = np.array([s.weight for s in scores], dtype=float)
    cal = np.array([s.calibration for s in scores], dtype=float)
    cal_bar = float(np.sum(weights * cal))

    # 2. Elicit each selected node (parallel). A node is retried to reach a
    #    conclusion; only a genuine hard failure (after retries) is flagged.
    target_nodes = _select_nodes(base_spec, framework.nodes)
    result_spec = NetworkSpec.from_pgmpy(base_spec.to_pgmpy())  # deep copy via round-trip
    node_results: dict[str, NodeElicitation] = {}
    elicited, skipped = [], []

    def _elicit_node_once(node_name: str) -> NodeElicitation:
        configs = _parent_configs(base_spec, node_name)
        states = list(base_spec.states[node_name])
        parents = list(base_spec.nodes[node_name].parents)
        roles = agents[0].client.propose_roles(node_name, states, parents, len(agents))
        per_expert: dict[str, dict[Config, list[float]]] = {}
        rationales: dict[str, str] = {}
        for agent, role in zip(agents, roles):
            columns, rationale = agent.client.node_cpt(node_name, states, parents, configs, role)
            per_expert[agent.name] = columns
            rationales[agent.name] = rationale

        # node-level correlation across agents (flattened over all columns)
        flat = np.array([[v for c in configs for v in per_expert[a.name][c]] for a in agents])
        corr = mean_pairwise_correlation(flat) if len(agents) >= 2 else 0.0

        mean_columns: dict[Config, list[float]] = {}
        kappa_raws: list[float] = []
        for cfg in configs:
            vectors = np.array([per_expert[a.name][cfg] for a in agents], dtype=float)
            mean_columns[cfg] = [float(x) for x in cooke_pool(vectors, weights)]
            contributing = vectors[weights > 0]
            if len(contributing) >= 2:
                kappa_raws.append(kappa_from_panel_spread(contributing, correlation=corr))
        node_kappa_raw = float(np.median(kappa_raws)) if kappa_raws else ladder.kappa_for("uncertain")
        # kappa level reflects panel agreement (spread, correlation-discounted);
        # calibration only gates the top level — it never forces "uncertain".
        level = ladder.snap(node_kappa_raw)
        if level == "tight" and cal_bar < TIGHT_CALIBRATION_FLOOR:
            level = "normal"
        return NodeElicitation(
            node=node_name,
            roles=roles,
            rationales=rationales,
            mean_columns=mean_columns,
            per_expert=per_expert,
            kappa=ladder.kappa_for(level),
            kappa_level=level,
        )

    # Each agent call self-escalates within the client (soft time limit -> 'conclude
    # now', reusing prior reasoning). The runner does not retry; a node that still
    # hard-fails is flagged for re-run, never silently bootstrapped.
    timeout = framework.effort.hard_timeout_s * (len(agents) + 1) + 30.0
    with ThreadPoolExecutor(max_workers=max(1, framework.effort.concurrency)) as pool:
        futures = {pool.submit(_elicit_node_once, n): n for n in target_nodes}
        done = 0
        for future, node_name in list(futures.items()):
            done += 1
            progress(f"Eliciting {node_name}", 0.1 + 0.85 * done / max(1, len(target_nodes)))
            try:
                node_results[node_name] = future.result(timeout=timeout)
                elicited.append(node_name)
            except (FutureTimeout, Exception) as exc:  # noqa: BLE001 - partial results by design
                skipped.append(node_name)
                node_results[node_name] = NodeElicitation(
                    node=node_name, roles=[], rationales={}, mean_columns={},
                    per_expert={}, kappa=0.0, kappa_level="uncertain", error=str(exc),
                )

    # 3. Overlay elicited columns + per-node kappa onto the result spec.
    for node_name in elicited:
        ne = node_results[node_name]
        node = result_spec.nodes[node_name]
        node.cpt = {cfg: list(vec) for cfg, vec in ne.mean_columns.items()}
        node.kappa = ne.kappa
        node.kappa_level = ne.kappa_level
    result_spec.validate()

    distinct_models = len({a.model for a in agents})
    correlation_note = (
        f"{len(agents)} agents across {distinct_models} base model(s); "
        f"roles chosen per node by the LLM; calibration weights are seed-based "
        f"(role-neutral). Single-model panels are correlated — diversity is limited."
    )
    progress("Done", 1.0)
    return ElicitationRun(
        run_id=run_id,
        created_at=created_at,
        framework_name=framework.name,
        n_agents=framework.n_agents,
        agent_names=[a.name for a in agents],
        models=sorted({a.model for a in agents}),
        expert_scores=[
            {"name": a.name, "model": a.model, "calibration": s.calibration, "information": s.information, "weight": s.weight}
            for a, s in zip(agents, scores)
        ],
        elicited_nodes=elicited,
        skipped_nodes=skipped,
        correlation_note=correlation_note,
        spec=result_spec,
        nodes=node_results,
        seeds=[{"id": s.id, "text": s.text, "realization": s.realization, "unit": s.unit} for s in seeds],
        seed_answers={a.name: seed_quantiles.get(a.name, {}) for a in agents},
    )


__all__ = ["Agent", "NodeElicitation", "ElicitationRun", "run_elicitation", "build_agents"]
