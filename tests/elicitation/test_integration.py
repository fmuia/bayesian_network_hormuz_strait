"""Tests for whole-network elicitation (Layer 4 backend), using the fake client."""

from __future__ import annotations

import numpy as np

from src.elicitation.integration import (
    EffortConfig,
    ElicitationFramework,
    ModelSpec,
    ScriptedClient,
    default_seeds,
    list_runs,
    load_run,
    run_elicitation,
    save_run,
)
from src.elicitation.integration.framework import list_frameworks, load_framework, save_framework
from src.network import build_network
from src.network_spec import NetworkSpec
from src.sensitivity import node_credible_intervals

SUBSET = ["US_Iran_Negotiations", "Iranian_Regime_Stability", "Iran_Aligned_Militia_Attacks", "Tanker_Incidents"]


def _node_fn(node, config, states):
    rng = np.random.default_rng(abs(hash((node, config))) % (2**32))
    return list(rng.dirichlet(np.ones(len(states)) * 2))


def _calibrated_client(name: str) -> ScriptedClient:
    answers = {s.id: (s.realization * 0.6, s.realization, s.realization * 1.4) for s in default_seeds()}
    return ScriptedClient(name, answers, _node_fn, rationale=f"{name} reasoning")


def _overconfident_client(name: str) -> ScriptedClient:
    answers = {s.id: (s.realization * 5, s.realization * 6, s.realization * 7) for s in default_seeds()}
    return ScriptedClient(name, answers, _node_fn, rationale=f"{name} reasoning")


def _factory(spec: ModelSpec) -> ScriptedClient:
    return _calibrated_client(spec.model) if spec.model == "calib" else _overconfident_client(spec.model)


def _framework(nodes=None, n_agents: int = 3) -> ElicitationFramework:
    return ElicitationFramework(
        name="test",
        models=[ModelSpec("scripted", "calib", "Calib"), ModelSpec("scripted", "over", "Over")],
        n_agents=n_agents,
        nodes=list(SUBSET if nodes is None else nodes),
        effort=EffortConfig(n_seeds=6, concurrency=2),
    )


def _run(framework=None):
    base = NetworkSpec.from_pgmpy(build_network())
    return base, run_elicitation(
        base, framework or _framework(), run_id="r1", created_at="2026-06-09T00:00:00Z",
        client_factory=_factory,
    )


def test_run_produces_valid_inference_ready_network() -> None:
    _, run = _run()
    run.spec.validate()
    run.spec.to_pgmpy().check_model()
    assert run.spec.kappa_map()  # elicited nodes carry per-CPT kappa


def test_selected_nodes_are_elicited_rest_stay_bootstrap() -> None:
    _, run = _run(_framework(nodes=SUBSET))
    assert set(run.elicited_nodes) == set(SUBSET)
    # unselected nodes are neither elicited nor skipped — they keep the bootstrap CPT
    for n in run.spec.nodes:
        if n not in SUBSET:
            assert run.spec.nodes[n].kappa is None
            assert n not in run.skipped_nodes


def test_empty_selection_elicits_the_whole_network() -> None:
    base = NetworkSpec.from_pgmpy(build_network())
    _, run = _run(_framework(nodes=[]))
    assert set(run.elicited_nodes) == set(base.nodes)


def test_agreeing_panel_is_not_forced_to_uncertain_by_low_calibration() -> None:
    """The kappa-always-uncertain bug: strong agreement must reach normal/tight
    even when seed calibration is ~0 (calibration only gates the top level)."""
    base = NetworkSpec.from_pgmpy(build_network())

    def fixed_node(node, config, states):
        return [0.2, 0.3, 0.5] if len(states) == 3 else [1.0 / len(states)] * len(states)

    def factory(spec):
        overconfident = {s.id: (s.realization * 5, s.realization * 6, s.realization * 7) for s in default_seeds()}
        return ScriptedClient(spec.model, overconfident, fixed_node)

    fw = ElicitationFramework(
        name="agree", models=[ModelSpec("scripted", "m", "M")], n_agents=3,
        nodes=["Tanker_Incidents"], effort=EffortConfig(n_seeds=6, concurrency=1),
    )
    run = run_elicitation(base, fw, run_id="r", created_at="t", client_factory=factory)
    assert run.nodes["Tanker_Incidents"].kappa_level in {"tight", "normal"}  # not "uncertain"


def test_run_records_seeds_and_agent_answers() -> None:
    _, run = _run(_framework(n_agents=2))
    assert len(run.seeds) == 6
    assert {s["id"] for s in run.seeds} == {s.id for s in default_seeds()}
    for agent in run.agent_names:
        assert set(run.seed_answers[agent]) == {s.id for s in default_seeds()}


def test_seed_scoring_distinguishes_calibrated_from_overconfident() -> None:
    _, run = _run(_framework(n_agents=2))  # agent-1=calib, agent-2=over
    by_model = {s["model"]: s for s in run.expert_scores}
    assert by_model["calib"]["calibration"] > by_model["over"]["calibration"]
    assert by_model["calib"]["weight"] > by_model["over"]["weight"]


def test_llm_recruits_roles_and_captures_rationales_per_node() -> None:
    _, run = _run(_framework(nodes=SUBSET[:3], n_agents=3))
    for node in run.elicited_nodes:
        ne = run.nodes[node]
        assert len(ne.roles) == 3
        assert len(ne.rationales) == 3
        assert ne.kappa_level in {"tight", "normal", "uncertain"}


def test_run_save_load_roundtrip(tmp_path) -> None:
    _, run = _run()
    save_run(run, tmp_path)
    assert len(list_runs(tmp_path)) == 1
    restored = load_run("r1", tmp_path)
    assert restored.elicited_nodes == run.elicited_nodes
    assert restored.spec.kappa_map() == run.spec.kappa_map()
    assert restored.seeds == run.seeds                 # the questions asked are preserved
    assert restored.seed_answers == run.seed_answers   # and each agent's answers
    for node in run.elicited_nodes:
        assert restored.nodes[node].roles == run.nodes[node].roles
        for cfg, vec in run.nodes[node].mean_columns.items():
            np.testing.assert_allclose(restored.nodes[node].mean_columns[cfg], vec)


def test_locked_network_runs_inference_with_per_cpt_kappa() -> None:
    _, run = _run()  # SUBSET includes Tanker_Incidents
    net = run.spec.to_pgmpy()
    km = run.spec.kappa_map()
    concentration = {v: km.get(v, 20.0) for v in net.nodes()}  # complete per-CPT map
    ci = node_credible_intervals(
        {"Iran_Aligned_Militia_Attacks": "high"},
        nodes=["Tanker_Incidents"],
        m=60,
        seed=0,
        base_network=net,
        concentration=concentration,
    )
    assert set(ci["Tanker_Incidents"]) == {"none", "isolated", "frequent"}


def test_framework_save_load_roundtrip(tmp_path) -> None:
    fw = _framework()
    save_framework(fw, tmp_path)
    assert "test" in list_frameworks(tmp_path)
    restored = load_framework("test", tmp_path)
    assert restored.n_agents == fw.n_agents
    assert restored.nodes == fw.nodes
    assert restored.effort.reasoning_effort == fw.effort.reasoning_effort
    assert restored.effort.time_limit_s == fw.effort.time_limit_s
    assert [m.model for m in restored.models] == [m.model for m in fw.models]


# --------------------------------------------------------------------------- #
# A node is never silently bootstrapped: a persistent hard failure is flagged.
# An empty seed set runs equal-weighted.
# --------------------------------------------------------------------------- #


class _FailingClient(ScriptedClient):
    def __init__(self, name: str) -> None:
        answers = {s.id: (s.realization * 0.6, s.realization, s.realization * 1.4) for s in default_seeds()}
        super().__init__(name, answers, _node_fn)

    def node_cpt(self, *args, **kwargs):
        raise RuntimeError("model unavailable")


def test_persistent_failure_is_flagged_not_silently_bootstrapped() -> None:
    base = NetworkSpec.from_pgmpy(build_network())
    fw = ElicitationFramework(
        name="fail", models=[ModelSpec("scripted", "x", "X")], n_agents=1,
        nodes=["Tanker_Incidents"], effort=EffortConfig(concurrency=1),
    )
    run = run_elicitation(base, fw, run_id="r", created_at="t", client_factory=lambda m: _FailingClient(m.model))
    assert "Tanker_Incidents" in run.skipped_nodes          # flagged for re-run
    assert "Tanker_Incidents" not in run.elicited_nodes
    assert run.nodes["Tanker_Incidents"].error is not None  # not a silent bootstrap
    assert run.spec.nodes["Tanker_Incidents"].kappa is None  # stays at bootstrap CPT


def test_empty_seed_set_runs_equal_weighted() -> None:
    base = NetworkSpec.from_pgmpy(build_network())
    fw = ElicitationFramework(
        name="noseed", models=[ModelSpec("scripted", "calib", "C")], n_agents=2,
        nodes=["Tanker_Incidents"], seeds=[], effort=EffortConfig(concurrency=1),
    )
    run = run_elicitation(base, fw, run_id="r", created_at="t", client_factory=_factory)
    assert "Tanker_Incidents" in run.elicited_nodes
    assert run.seeds == []
    assert all(abs(s["weight"] - 0.5) < 1e-9 for s in run.expert_scores)  # equal weighting
    assert all(s["calibration"] == 0.0 for s in run.expert_scores)        # no calibration evidence


def test_seed_set_save_load_roundtrip(tmp_path) -> None:
    from src.elicitation.integration import load_seeds, save_seeds
    from src.elicitation.protocols.base import SeedQuestion

    seeds = [SeedQuestion("q1", "How many X?", 42.0, "count"), SeedQuestion("q2", "What share?", 0.2, "%")]
    path = tmp_path / "seeds.json"
    save_seeds(seeds, path)
    restored = load_seeds(path)
    assert [s.text for s in restored] == ["How many X?", "What share?"]
    assert restored[0].realization == 42.0
    assert load_seeds(tmp_path / "nonexistent.json") == []


def test_urgent_prompt_reuses_prior_reasoning() -> None:
    from src.elicitation.integration.clients import URGENT_SUFFIX, build_urgent_prompt

    with_partial = build_urgent_prompt("Estimate X.", "I weighed factors A and B.")
    assert "I weighed factors A and B." in with_partial  # prior reasoning is carried, not wasted
    assert URGENT_SUFFIX in with_partial
    without_partial = build_urgent_prompt("Estimate X.", "   ")
    assert "reasoning so far" not in without_partial
    assert URGENT_SUFFIX in without_partial
