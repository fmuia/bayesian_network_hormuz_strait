"""Sanity tests for the BN definition, inference, and evidence catalogue."""

from __future__ import annotations

import math

import pytest

from src.evidence import EVENTS
from src.inference import BNInferenceEngine
from src.network import EDGES, STATES, build_network


def test_network_builds_and_validates() -> None:
    net = build_network()
    assert net.check_model() is True
    # No orphan nodes (every state-vocab node is in the graph).
    assert set(net.nodes()) == set(STATES.keys())
    # Edge list matches the declared structure.
    assert set(net.edges()) == set(EDGES)


def test_all_cpts_normalised() -> None:
    net = build_network()
    for cpd in net.get_cpds():
        values = cpd.get_values()
        # Each column (one per parent assignment) sums to 1.
        for col in range(values.shape[1]):
            assert math.isclose(values[:, col].sum(), 1.0, abs_tol=1e-8), (
                f"CPT for {cpd.variable} column {col} sums to {values[:, col].sum()}"
            )


def test_prior_scenario_distribution() -> None:
    engine = BNInferenceEngine()
    prior = engine.get_prior_probabilities()
    assert set(prior.keys()) == set(STATES["Scenario"])
    assert math.isclose(sum(prior.values()), 1.0, abs_tol=1e-8)
    # Sanity: no scenario degenerate at zero or one.
    for p in prior.values():
        assert 0.0 < p < 1.0


def test_extreme_escalation_raises_severe_closure() -> None:
    engine = BNInferenceEngine()
    prior_severe = engine.get_prior_probabilities()["Severe_Closure"]
    engine.update_evidence({
        "US_Iran_Negotiations": "breakdown",
        "Sanctions_Trajectory": "tightening",
        "Iranian_Regime_Stability": "unstable",
        "Tanker_Incidents": "frequent",
        "US_Military_Response": "major",
    })
    posterior_severe = engine.get_scenario_probabilities()["Severe_Closure"]
    assert posterior_severe > prior_severe + 0.10  # material shift


def test_extreme_deescalation_raises_stress_mitigates() -> None:
    engine = BNInferenceEngine()
    prior_mit = engine.get_prior_probabilities()["Stress_Mitigates"]
    engine.update_evidence({
        "US_Iran_Negotiations": "success",
        "Third_Party_Mediation": "active",
        "Iranian_Regime_Stability": "stable",
        "Tanker_Incidents": "none",
        "US_Military_Response": "none",
    })
    posterior_mit = engine.get_scenario_probabilities()["Stress_Mitigates"]
    assert posterior_mit > prior_mit + 0.05


def test_clear_evidence_returns_to_prior() -> None:
    engine = BNInferenceEngine()
    prior = engine.get_prior_probabilities()
    engine.update_evidence({"US_Iran_Negotiations": "breakdown"})
    engine.clear_evidence()
    after = engine.get_scenario_probabilities()
    for k in prior:
        assert math.isclose(prior[k], after[k], abs_tol=1e-8)


def test_node_marginal_observed_node_is_delta() -> None:
    engine = BNInferenceEngine()
    engine.update_evidence({"Strait_Operationally_Closed": "full"})
    m = engine.get_node_marginal("Strait_Operationally_Closed")
    assert math.isclose(m["full"], 1.0, abs_tol=1e-8)
    assert math.isclose(m["no"] + m["partial"], 0.0, abs_tol=1e-8)


@pytest.mark.parametrize("event", EVENTS)
def test_evidence_catalogue_well_formed(event) -> None:
    """Every event references valid nodes and valid states."""
    assert event.id and event.headline and event.date
    assert event.category in {"escalation", "de-escalation", "mixed"}
    for node, state in event.assignments.items():
        assert node in STATES, f"unknown node {node} in event {event.id}"
        assert state in STATES[node], (
            f"invalid state {state!r} for node {node} in event {event.id}"
        )


def test_evidence_events_apply_via_engine() -> None:
    engine = BNInferenceEngine()
    for event in EVENTS:
        engine.clear_evidence()
        engine.update_evidence(event.assignments)
        probs = engine.get_scenario_probabilities()
        assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-6)
