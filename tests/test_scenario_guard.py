"""Scenario is the latent-regime query target — evidence on it must be ignored
(dropped), not crash the Scenario-targeted queries (engine marginal + CI).

Regression for: pgmpy "Can't have the same variables in both `variables` and
`evidence`. Found in both: {'Scenario'}".
"""
from __future__ import annotations

from src.inference import BNInferenceEngine
from src.network import STATES, build_network
from src.sensitivity import scenario_credible_intervals


def test_engine_scenario_probs_ignore_scenario_evidence():
    eng = BNInferenceEngine(build_network("latent_regime"))
    eng.update_evidence({"Scenario": "Severe_Closure", "Tanker_Incidents": "frequent"})
    probs = eng.get_scenario_probabilities()                 # must not raise
    assert set(probs) == set(STATES["Scenario"])
    assert abs(sum(probs.values()) - 1.0) < 1e-6             # inferred posterior, not a delta


def test_scenario_credible_intervals_ignore_scenario_evidence():
    ci = scenario_credible_intervals(
        {"Scenario": "Severe_Closure", "Tanker_Incidents": "frequent"},
        m=20, concentration=20, base_network=build_network("latent_regime"),
    )                                                        # must not raise
    assert set(ci) == set(STATES["Scenario"])
    for _mean, lo, hi in ci.values():
        assert 0.0 <= lo <= hi <= 1.0
