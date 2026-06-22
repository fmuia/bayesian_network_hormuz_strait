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
    probe_call,
    project_runtime,
    run_elicitation,
    save_run,
)
from src.elicitation.integration.framework import list_frameworks, load_framework, save_framework
from src.scenario import build_network
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


def _recall_factory(recall_ids):
    """Calibrated scripted clients that self-report the given seed ids as 'recall'."""
    def factory(spec):
        answers = {s.id: (s.realization * 0.6, s.realization, s.realization * 1.4) for s in default_seeds()}
        return ScriptedClient(spec.model, answers, _node_fn, recall_ids=set(recall_ids))
    return factory


def test_default_basis_keeps_all_seeds() -> None:
    _, run = _run()  # scripted clients report 'estimate' for everything
    c = run.contamination
    assert c is not None
    assert c["discarded_seeds"] == []
    assert c["n_seeds_scored"] == c["n_seeds_asked"] == len(default_seeds())
    assert c["any_flagged"] is False


def test_recalled_seeds_are_discarded_from_scoring() -> None:
    base = NetworkSpec.from_pgmpy(build_network())
    recalled = {s.id for s in default_seeds()[:3]}  # a majority-recalled subset
    fw = _framework(nodes=["Tanker_Incidents"], n_agents=2)
    run = run_elicitation(base, fw, run_id="r", created_at="t", client_factory=_recall_factory(recalled))
    c = run.contamination
    assert set(c["discarded_seeds"]) == recalled
    assert c["n_seeds_scored"] == len(default_seeds()) - 3
    assert c["any_flagged"] is True
    # the discarded seeds still appear in the asked-seed record (transparency)
    assert {s["id"] for s in run.seeds} == {s.id for s in default_seeds()}


def test_all_recalled_falls_back_to_equal_weighting() -> None:
    base = NetworkSpec.from_pgmpy(build_network())
    all_ids = {s.id for s in default_seeds()}
    fw = _framework(nodes=["Tanker_Incidents"], n_agents=2)
    run = run_elicitation(base, fw, run_id="r", created_at="t", client_factory=_recall_factory(all_ids))
    assert run.contamination["n_seeds_scored"] == 0
    assert set(run.contamination["discarded_seeds"]) == all_ids
    assert all(abs(s["weight"] - 0.5) < 1e-9 for s in run.expert_scores)   # equal weighting
    assert all(s["calibration"] == 0.0 for s in run.expert_scores)         # no calibration evidence


def test_contamination_summary_survives_save_load(tmp_path) -> None:
    _, run = _run()
    save_run(run, tmp_path)
    assert load_run("r1", tmp_path).contamination == run.contamination


def test_parallel_agents_give_same_result_regardless_of_concurrency() -> None:
    """Agents now fan out under a global concurrency cap; the pooled result must
    be identical whether one call runs at a time or several."""
    base = NetworkSpec.from_pgmpy(build_network())

    def run_at(concurrency: int):
        fw = ElicitationFramework(
            name="c", models=[ModelSpec("scripted", "calib", "C")], n_agents=3,
            nodes=list(SUBSET), effort=EffortConfig(n_seeds=6, concurrency=concurrency),
        )
        return run_elicitation(base, fw, run_id="r", created_at="t", client_factory=_factory)

    serial, parallel = run_at(1), run_at(8)
    assert set(serial.elicited_nodes) == set(parallel.elicited_nodes)
    for node in serial.elicited_nodes:
        for cfg, vec in serial.nodes[node].mean_columns.items():
            np.testing.assert_allclose(parallel.nodes[node].mean_columns[cfg], vec)
        assert serial.nodes[node].kappa == parallel.nodes[node].kappa


def test_run_captures_timing_diagnostics() -> None:
    _, run = _run(_framework(nodes=SUBSET, n_agents=3))
    diag = run.diagnostics
    assert diag is not None
    # seeds(3) + roles(4 nodes) + cpts(4 nodes * 3 agents) = 19 calls
    assert diag["n_calls"] == 3 + len(SUBSET) + len(SUBSET) * 3
    assert diag["n_failed"] == 0
    assert set(diag["by_phase"]) == {"seed", "roles", "cpt"}
    assert diag["wall_s"] >= 0.0


def test_diagnostics_survive_save_load(tmp_path) -> None:
    _, run = _run()
    save_run(run, tmp_path)
    restored = load_run("r1", tmp_path)
    assert restored.diagnostics == run.diagnostics


def test_project_runtime_is_work_conserving() -> None:
    # 12 nodes, 3 agents, concurrency 3, seeds on, uniform 10s calls:
    # calls = 3 + 12 + 12*3 = 51 ; work = 51*10 = 510 ; wall = 510/3 = 170
    p = project_runtime(n_nodes=12, n_agents=3, concurrency=3, cpt_s=10.0, has_seeds=True)
    assert p["total_calls"] == 51
    assert p["work_s"] == 510.0
    assert p["est_wall_s"] == 170.0
    # slow roles calls are costed separately, not applied to every call:
    # work = (3+36)*19 + 12*47 = 741 + 564 = 1305 ; wall = 1305/4 = 326.25
    p2 = project_runtime(12, 3, 4, cpt_s=19.0, roles_s=47.0, has_seeds=True)
    assert p2["work_s"] == 1305.0
    assert p2["est_wall_s"] == 326.2
    # no seeds drops the seed-scoring calls
    assert project_runtime(12, 3, 3, 10.0, has_seeds=False)["total_calls"] == 48


def test_probe_call_times_one_call_and_projects(tmp_path) -> None:
    base = NetworkSpec.from_pgmpy(build_network())
    fw = _framework(nodes=SUBSET, n_agents=3)
    p = probe_call(base, fw, client_factory=_factory)
    assert p["error"] is None
    assert p["node"] in SUBSET
    assert p["total_calls"] == len(SUBSET) * (1 + 3) + 3  # seeds counted (framework has seeds)
    assert p["est_wall_s"] >= 0.0


def test_urgent_prompt_reuses_prior_reasoning() -> None:
    from src.elicitation.integration.clients import URGENT_SUFFIX, build_urgent_prompt

    with_partial = build_urgent_prompt("Estimate X.", "I weighed factors A and B.")
    assert "I weighed factors A and B." in with_partial  # prior reasoning is carried, not wasted
    assert URGENT_SUFFIX in with_partial
    without_partial = build_urgent_prompt("Estimate X.", "   ")
    assert "reasoning so far" not in without_partial
    assert URGENT_SUFFIX in without_partial
