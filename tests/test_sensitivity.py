"""Tests for CPT-resampling-based credible intervals."""

from __future__ import annotations

import math

from src.inference import BNInferenceEngine
from src.scenario import STATES
from src.sensitivity import (
    node_credible_intervals,
    scenario_credible_intervals,
)


def test_node_ci_matches_scenario_ci_for_scenario_node() -> None:
    """node_credible_intervals(nodes=["Scenario"]) must agree numerically
    with the scenario-specific function under identical seed and params."""
    scenario_only = scenario_credible_intervals(
        {}, m=40, concentration=20.0, seed=0
    )
    via_nodes = node_credible_intervals(
        {}, nodes=["Scenario"], m=40, concentration=20.0, seed=0
    )["Scenario"]
    for s in STATES["Scenario"]:
        mean_a, lo_a, hi_a = scenario_only[s]
        mean_b, lo_b, hi_b = via_nodes[s]
        assert math.isclose(mean_a, mean_b, abs_tol=1e-9)
        assert math.isclose(lo_a, lo_b, abs_tol=1e-9)
        assert math.isclose(hi_a, hi_b, abs_tol=1e-9)


def test_hard_observed_node_returns_delta() -> None:
    out = node_credible_intervals(
        {"Strait_Operationally_Closed": "full"},
        nodes=["Strait_Operationally_Closed"],
        m=5,
    )["Strait_Operationally_Closed"]
    assert out["full"] == (1.0, 1.0, 1.0)
    assert out["no"] == (0.0, 0.0, 0.0)
    assert out["partial"] == (0.0, 0.0, 0.0)


def test_ci_brackets_point_estimate_marginal() -> None:
    """For an unobserved node, the (lo, hi) CI should bracket the point-
    estimate posterior marginal (or at least include the mean sample)."""
    engine = BNInferenceEngine()
    engine.update_evidence({"Tanker_Incidents": "frequent"})
    point = engine.get_node_marginal("Scenario")

    ci = node_credible_intervals(
        {"Tanker_Incidents": "frequent"},
        nodes=["Scenario"],
        m=200,
        concentration=20.0,
        seed=0,
    )["Scenario"]
    for state, p in point.items():
        mean, lo, hi = ci[state]
        # The Dirichlet-resampled mean should be near the point estimate.
        assert abs(mean - p) < 0.05, (state, mean, p)
        # The CI should bracket the mean by construction.
        assert lo <= mean <= hi


def test_all_nodes_by_default_returns_every_node() -> None:
    out = node_credible_intervals({}, m=5)
    assert set(out.keys()) == set(STATES.keys())
    for node, dist in out.items():
        assert set(dist.keys()) == set(STATES[node])


def test_soft_evidence_applied_when_present() -> None:
    """Passing soft evidence on an upstream node should shift downstream
    posteriors relative to the no-evidence baseline."""
    baseline = node_credible_intervals(
        {}, nodes=["Scenario"], m=80, seed=0
    )["Scenario"]
    with_soft = node_credible_intervals(
        {},
        soft_evidence={
            "Tanker_Incidents": {
                "none": 0.05, "isolated": 0.15, "frequent": 0.80
            }
        },
        nodes=["Scenario"],
        m=80,
        seed=0,
    )["Scenario"]
    # Severe_Closure should get more mass under a "mostly frequent" prior.
    assert with_soft["Severe_Closure"][0] > baseline["Severe_Closure"][0]


def test_invalid_node_raises() -> None:
    import pytest
    with pytest.raises(KeyError):
        node_credible_intervals({}, nodes=["Not_A_Node"], m=2)


def test_invalid_ci_raises() -> None:
    import pytest
    with pytest.raises(ValueError):
        node_credible_intervals({}, nodes=["Scenario"], m=2, ci=1.5)
