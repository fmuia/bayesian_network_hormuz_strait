"""Whole-network elicitation runner.

Recruits ``n_agents`` agents distributed across the framework's selected models,
scores them on the seeds once, then for each in-scope node: lets the LLM pick
node-appropriate roles, elicits the node's CPT from each agent (one call per
agent per node), and pools the columns with the Cooke weights — estimating a
per-node kappa from the panel's spread (correlation-discounted) and capping it
by the panel's calibration.

Scheduling. Every LLM call (seed scoring, role proposal, and each agent's CPT)
is a leaf task dispatched onto a *single* pool whose width is ``concurrency`` —
so ``concurrency`` is the number of LLM calls in flight at once, across all nodes
*and* agents. Agents within a node no longer run one-after-another; a node's
agent calls fan out and compete for the same global slots as every other node's.

Effort controls bound cost/time: cheap node selection, bounded concurrency,
per-call soft/hard timeouts, and partial results (a node whose roles or any
agent call hard-fails is skipped and keeps its bootstrap CPT). Per-call wall
times are captured into :attr:`ElicitationRun.diagnostics` so a run's timing can
be checked against the configured budget.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np

from ...network_spec import NetworkSpec
from ..engine.aggregation import cooke_pool
from ..engine.calibration import DEFAULT_QUANTILES, ExpertScore, classical_model_weights
from ..engine.kappa import KappaLadder, kappa_from_panel_spread
from ..agents.contamination import source_attribution_probe, summarize_probes
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
class CallRecord:
    """Wall-clock for one LLM call, for timing diagnostics."""

    phase: str            # "seed" | "roles" | "cpt"
    duration_s: float
    escalated: bool       # ran past the soft time budget (was asked to conclude)
    failed: bool
    node: str | None = None
    agent: str | None = None


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
    diagnostics: dict | None = None                             # timing summary (see _summarize_timings)
    contamination: dict | None = None                           # source-attribution probe summary (see runner §1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _run_call(fn: Callable, *args):
    """Run one leaf call, returning ``(result, duration_s, error)``. Never raises
    — a failed call is reported via the error slot so the pool keeps draining."""
    t0 = time.perf_counter()
    try:
        return fn(*args), time.perf_counter() - t0, None
    except Exception as exc:  # noqa: BLE001 - reported, not raised (partial results by design)
        return None, time.perf_counter() - t0, exc


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


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), p))


def _summarize_timings(records: list[CallRecord], wall_s: float, soft_s: float, hard_s: float) -> dict:
    """Aggregate per-call timings into a compact diagnostics dict for display."""
    by_phase: dict[str, dict] = {}
    for phase in ("seed", "roles", "cpt"):
        durs = [r.duration_s for r in records if r.phase == phase]
        if durs:
            by_phase[phase] = {
                "count": len(durs),
                "p50_s": round(_percentile(durs, 50), 2),
                "p95_s": round(_percentile(durs, 95), 2),
                "max_s": round(max(durs), 2),
            }
    slowest = max(records, key=lambda r: r.duration_s, default=None)
    return {
        "wall_s": round(wall_s, 1),
        "soft_limit_s": soft_s,
        "hard_timeout_s": hard_s,
        "n_calls": len(records),
        "n_escalated": sum(r.escalated for r in records),  # hit the soft budget → 'conclude now'
        "n_failed": sum(r.failed for r in records),
        "by_phase": by_phase,
        "slowest": (
            {
                "phase": slowest.phase,
                "node": slowest.node,
                "agent": slowest.agent,
                "duration_s": round(slowest.duration_s, 2),
            }
            if slowest
            else None
        ),
    }


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
    concurrency = max(1, framework.effort.concurrency)
    soft_s = framework.effort.time_limit_s
    hard_s = framework.effort.hard_timeout_s

    records: list[CallRecord] = []

    def record(phase: str, duration: float, err: Exception | None, node=None, agent=None) -> None:
        records.append(
            CallRecord(
                phase=phase,
                duration_s=duration,
                escalated=(err is None and duration >= soft_s),
                failed=err is not None,
                node=node,
                agent=agent,
            )
        )

    wall_t0 = time.perf_counter()

    target_nodes = _select_nodes(base_spec, framework.nodes)
    states = {n: list(base_spec.states[n]) for n in target_nodes}
    parents = {n: list(base_spec.nodes[n].parents) for n in target_nodes}
    configs = {n: _parent_configs(base_spec, n) for n in target_nodes}

    # A generous wall-clock guard against a genuinely wedged call. Each leaf call
    # is already bounded by the client's hard timeout; this only catches a future
    # that never returns at all.
    total_calls = (len(agents) if seeds else 0) + len(target_nodes) * (1 + len(agents))
    waves = max(1, math.ceil(total_calls / concurrency))
    overall_guard = hard_s * (waves + 2) + 60.0

    def _equal_scores() -> list[ExpertScore]:
        return [ExpertScore(0.0, 0.0, 1.0, 1.0 / len(agents)) for _ in agents]

    contamination: dict | None = None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        # 1. Score the panel on the seeds (role-neutral), in parallel. Seeds the
        #    panel admits it *recalled* (source-attribution self-report) are
        #    discarded before scoring — they test memory, not calibration (§8.3).
        #    With no seeds (or all discarded), fall back to equal weighting.
        seed_quantiles: dict[str, dict[str, list[float]]] = {}
        seed_basis: dict[str, dict[str, str]] = {}
        if seeds:
            progress("Scoring agents on seeds", 0.05)
            seed_futs = {
                pool.submit(_run_call, a.client.seed_quantiles, seeds, levels, None): a for a in agents
            }
            for f in as_completed(seed_futs):
                agent = seed_futs[f]
                out, dur, err = f.result()
                record("seed", dur, err, agent=agent.name)
                seed_quantiles[agent.name] = ({} if err else out[0])
                seed_basis[agent.name] = ({} if err else out[1])

            # source-attribution: drop a seed the panel mostly recalls (majority vote)
            recall_rate = {
                s.id: float(np.mean([1.0 if seed_basis.get(a.name, {}).get(s.id) == "recall" else 0.0
                                     for a in agents]))
                for s in seeds
            }
            scored_seeds = [s for s in seeds if recall_rate[s.id] < 0.5]
            probes = [
                source_attribution_probe(f"{a.name}:{s.id}",
                                         seed_basis.get(a.name, {}).get(s.id) == "recall")
                for a in agents for s in seeds
            ]
            contamination = {
                "method": "source-attribution self-report; seeds recalled by a majority are discarded",
                "n_seeds_asked": len(seeds),
                "n_seeds_scored": len(scored_seeds),
                "discarded_seeds": [s.id for s in seeds if recall_rate[s.id] >= 0.5],
                "recall_rate": {sid: round(r, 2) for sid, r in recall_rate.items()},
                **summarize_probes(probes),
            }
            if scored_seeds:
                experts_quantiles = [
                    np.array([seed_quantiles[a.name].get(s.id, [0.0] * len(levels)) for s in scored_seeds],
                             dtype=float)
                    for a in agents
                ]
                realizations = np.array([s.realization for s in scored_seeds], dtype=float)
                scores = classical_model_weights(experts_quantiles, realizations, levels, alpha=0.0)
            else:
                scores = _equal_scores()  # every seed was recalled — no calibration evidence left
        else:
            scores = _equal_scores()
        weights = np.array([s.weight for s in scores], dtype=float)
        cal = np.array([s.calibration for s in scores], dtype=float)
        cal_bar = float(np.sum(weights * cal))

        # 2. Roles per node, then each node's agent CPTs — all leaf calls share the
        #    same `concurrency` slots. Agents within a node run in parallel; a node's
        #    CPT calls are submitted the moment its roles return.
        roles_by_node: dict[str, list[str]] = {}
        agent_futs: dict = {}                                   # future -> (node, agent_idx)
        agent_cols: dict[str, dict[str, dict[Config, list[float]]]] = defaultdict(dict)
        agent_rats: dict[str, dict[str, str]] = defaultdict(dict)
        node_failed: dict[str, bool] = defaultdict(bool)

        roles_futs = {
            pool.submit(_run_call, agents[0].client.propose_roles, n, states[n], parents[n], len(agents)): n
            for n in target_nodes
        }
        progress("Recruiting roles & eliciting CPTs", 0.1)
        try:
            for f in as_completed(roles_futs, timeout=overall_guard):
                node = roles_futs[f]
                out, dur, err = f.result()
                record("roles", dur, err, node=node)
                if err is not None:
                    node_failed[node] = True
                    continue
                roles_by_node[node] = out
                for idx, agent in enumerate(agents):
                    role = out[idx] if idx < len(out) else None
                    af = pool.submit(
                        _run_call, agent.client.node_cpt, node, states[node], parents[node], configs[node], role
                    )
                    agent_futs[af] = (node, idx)
        except FutureTimeout:
            pass
        for f, node in roles_futs.items():
            if not f.done():
                f.cancel()
                node_failed[node] = True
                record("roles", overall_guard, TimeoutError("roles call did not return"), node=node)

        done = 0
        try:
            for f in as_completed(list(agent_futs), timeout=overall_guard):
                node, idx = agent_futs[f]
                agent = agents[idx]
                out, dur, err = f.result()
                record("cpt", dur, err, node=node, agent=agent.name)
                done += 1
                progress(f"Eliciting {node}", 0.1 + 0.85 * done / max(1, len(agent_futs)))
                if err is not None:
                    node_failed[node] = True
                else:
                    columns, rationale = out
                    agent_cols[node][agent.name] = columns
                    agent_rats[node][agent.name] = rationale
        except FutureTimeout:
            pass
        for f, (node, idx) in agent_futs.items():
            if not f.done():
                f.cancel()
                node_failed[node] = True
                record("cpt", overall_guard, TimeoutError("CPT call did not return"),
                       node=node, agent=agents[idx].name)

    # 3. Aggregate each node whose roles and all agent calls succeeded. A node
    #    that hard-failed anywhere is flagged for re-run, never silently bootstrapped.
    result_spec = NetworkSpec.from_pgmpy(base_spec.to_pgmpy())  # deep copy via round-trip
    node_results: dict[str, NodeElicitation] = {}
    elicited, skipped = [], []
    for node in target_nodes:
        cols = agent_cols.get(node, {})
        if node not in roles_by_node or node_failed[node] or len(cols) < len(agents):
            skipped.append(node)
            node_results[node] = NodeElicitation(
                node=node, roles=roles_by_node.get(node, []), rationales={}, mean_columns={},
                per_expert={}, kappa=0.0, kappa_level="uncertain",
                error="role or CPT call failed — re-run this node",
            )
            continue
        node_results[node] = _aggregate_node(
            node, roles_by_node[node], cols, agent_rats[node],
            configs[node], agents, weights, cal_bar, ladder,
        )
        elicited.append(node)

    # 4. Overlay elicited columns + per-node kappa onto the result spec.
    for node_name in elicited:
        ne = node_results[node_name]
        node = result_spec.nodes[node_name]
        node.cpt = {cfg: list(vec) for cfg, vec in ne.mean_columns.items()}
        node.kappa = ne.kappa
        node.kappa_level = ne.kappa_level
    result_spec.validate()

    distinct_models = len({a.model for a in agents})
    diversity = (
        "Single base model — agents are correlated; add a second model for genuine diversity."
        if distinct_models < 2
        else f"{distinct_models} base models — partial genuine diversity."
    )
    correlation_note = (
        f"{len(agents)} agents across {distinct_models} base model(s). "
        f"Precautions taken: agents consulted independently (no cross-agent anchoring → counters "
        f"sycophancy); roles include an adversarial skeptic; prompts enforce outside-view base rates, "
        f"consider-the-opposite, reserve-mass-for-surprise, and reason-don't-recall; κ is discounted for "
        f"inter-agent correlation and capped by seed calibration. Weights are seed-based and role-neutral "
        f"— roles diversify the considerations behind the pooled mean and spread, not separately-calibrated "
        f"estimators (methodology §8.2). {diversity}"
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
        diagnostics=_summarize_timings(records, time.perf_counter() - wall_t0, soft_s, hard_s),
        contamination=contamination,
    )


def _aggregate_node(
    node: str,
    roles: list[str],
    per_expert: dict[str, dict[Config, list[float]]],
    rationales: dict[str, str],
    configs: list[Config],
    agents: list[Agent],
    weights: np.ndarray,
    cal_bar: float,
    ladder: KappaLadder,
) -> NodeElicitation:
    """Pool the panel's columns (Cooke weights) and estimate the node's kappa from
    the panel spread (correlation-discounted); calibration only gates the top level."""
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
    level = ladder.snap(node_kappa_raw)
    if level == "tight" and cal_bar < TIGHT_CALIBRATION_FLOOR:
        level = "normal"
    return NodeElicitation(
        node=node,
        roles=roles,
        rationales=rationales,
        mean_columns=mean_columns,
        per_expert=per_expert,
        kappa=ladder.kappa_for(level),
        kappa_level=level,
    )


# --------------------------------------------------------------------------- #
# Latency probe — sanity-check timing before committing to a whole-network run
# --------------------------------------------------------------------------- #


def _cheapest_node(spec: NetworkSpec, selected: list[str]) -> str:
    """The in-scope node with the fewest parent configurations (cheapest to probe)."""
    candidates = _select_nodes(spec, selected)
    return min(candidates, key=lambda n: len(_parent_configs(spec, n)))


def project_runtime(
    n_nodes: int,
    n_agents: int,
    concurrency: int,
    cpt_s: float,
    has_seeds: bool,
    roles_s: float | None = None,
) -> dict:
    """Project whole-network wall-clock from measured per-call latencies.

    The pool is work-conserving, so wall-clock ≈ total work-seconds / concurrency
    (not ``waves × slowest`` — that wrongly assumes every call is as slow as the
    slowest). Roles calls are usually slower than CPT calls, so they are costed
    separately; seed-scoring calls are costed like CPT calls.

        work_s = (seeds + cpts) · cpt_s  +  roles · roles_s
        wall   ≈ work_s / concurrency                     (+ tail/imbalance)
    """
    roles_s = cpt_s if roles_s is None else roles_s
    seed_calls = n_agents if has_seeds else 0
    roles_calls = n_nodes
    cpt_calls = n_nodes * n_agents
    total = seed_calls + roles_calls + cpt_calls
    work_s = (seed_calls + cpt_calls) * cpt_s + roles_calls * roles_s
    return {
        "total_calls": total,
        "work_s": round(work_s, 1),
        "est_wall_s": round(work_s / max(1, concurrency), 1),
    }


def probe_call(
    base_spec: NetworkSpec,
    framework: ElicitationFramework,
    node: str | None = None,
    client_factory: Callable[[ModelSpec], ElicitationClient] | None = None,
) -> dict:
    """Time ONE real roles + CPT call on the framework's first model, then project
    the full-network wall-clock for the current effort settings. Cheap way to check
    whether the configured time budget is realistic before launching a whole run."""
    if not framework.models:
        raise ValueError("framework has no models selected")
    model = framework.models[0]
    factory = client_factory or (
        lambda m: make_client(
            m.provider, m.model,
            framework.effort.time_limit_s, framework.effort.hard_timeout_s, framework.effort.reasoning_effort,
        )
    )
    client = factory(model)
    node = node or _cheapest_node(base_spec, framework.nodes)
    sts = list(base_spec.states[node])
    pts = list(base_spec.nodes[node].parents)
    cfgs = _parent_configs(base_spec, node)

    roles_out, t_roles, e_roles = _run_call(client.propose_roles, node, sts, pts, 1)
    role = roles_out[0] if (roles_out and not e_roles) else None
    _, t_cpt, e_cpt = _run_call(client.node_cpt, node, sts, pts, cfgs, role)

    n_nodes = len(_select_nodes(base_spec, framework.nodes))
    projection = project_runtime(
        n_nodes, framework.n_agents, framework.effort.concurrency,
        cpt_s=t_cpt, roles_s=t_roles,
        has_seeds=bool(framework.seeds),  # the runner scores on seeds whenever any are set
    )
    err = e_roles or e_cpt
    return {
        "node": node,
        "model": model.model,
        "roles_s": round(t_roles, 2),
        "cpt_s": round(t_cpt, 2),
        "roles_escalated": (e_roles is None and t_roles >= framework.effort.time_limit_s),
        "cpt_escalated": (e_cpt is None and t_cpt >= framework.effort.time_limit_s),
        "soft_limit_s": framework.effort.time_limit_s,
        "error": str(err) if err else None,
        **projection,
    }


__all__ = [
    "Agent",
    "CallRecord",
    "NodeElicitation",
    "ElicitationRun",
    "run_elicitation",
    "build_agents",
    "probe_call",
    "project_runtime",
]
