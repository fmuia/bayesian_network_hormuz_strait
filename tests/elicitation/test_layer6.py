"""Tests for Layer 6: confidence, prioritisation, calibration tiers, proposals."""

from __future__ import annotations

import numpy as np
import pytest

from src.elicitation.agents import ScriptedCompletionClient
from src.elicitation.calibration import (
    Tier2Tracker,
    brier_score,
    log_bayes_factor_trajectory,
    performance_from_brier,
    update_weights,
)
from src.elicitation.proposals import AnalogEvent, propose_cpt
from src.elicitation.protocols import CPTColumnTarget
from src.elicitation.reporting import (
    ConfidenceReport,
    assemble_confidence,
    ipcc_confidence,
    ipcc_likelihood,
)
from src.elicitation.sensitivity import run_prioritisation

# --------------------------------------------------------------------------- #
# Confidence reporting
# --------------------------------------------------------------------------- #


def test_ipcc_likelihood_language() -> None:
    assert ipcc_likelihood(0.999) == "virtually certain"
    assert ipcc_likelihood(0.70) == "likely"
    assert ipcc_likelihood(0.50) == "about as likely as not"
    assert ipcc_likelihood(0.02) == "very unlikely"
    with pytest.raises(ValueError):
        ipcc_likelihood(1.5)


def test_ipcc_confidence_matrix() -> None:
    assert ipcc_confidence("robust", "high") == "very high"
    assert ipcc_confidence("limited", "low") == "very low"
    assert ipcc_confidence("medium", "medium") == "medium"


def test_confidence_report_is_a_vector_not_a_scalar() -> None:
    report = assemble_confidence(
        query="P(crisis | E)",
        point=0.70,
        credible_interval=(0.55, 0.82),
        variance_shares={"P(T|S)": 0.68, "prior(S)": 0.20, "P(D|S)": 0.12},
        evidence="medium",
        agreement="medium",
    )
    assert isinstance(report, ConfidenceReport)
    # all four components are present
    assert report.point == 0.70
    assert report.credible_interval == (0.55, 0.82)
    assert report.likelihood == "likely"
    assert report.confidence == "medium"
    assert report.variance_shares["P(T|S)"] == 0.68
    text = report.summary()
    assert "conditional on model structure" in text.lower()
    assert "P(T|S)" in text  # the dominant driver is surfaced
    assert "model-internal" in text  # honest about no empirical validation yet


# --------------------------------------------------------------------------- #
# Prioritisation
# --------------------------------------------------------------------------- #


def test_prioritisation_ranks_dominant_cpt_first_and_recommends_tightening() -> None:
    cpts = {"A": ([0.5, 0.5], 6.0), "B": ([0.5, 0.5], 500.0)}

    def output(sample):
        return float(sample["A"][0] + 0.01 * sample["B"][0])

    priorities = run_prioritisation(
        cpts, output, kappa_levels={"A": "uncertain", "B": "tight"}, n_samples=1500, seed=0
    )
    assert priorities[0].node == "A"
    assert priorities[0].variance_share > priorities[1].variance_share
    assert priorities[0].recommended is True   # dominant + loose -> tighten
    assert priorities[1].recommended is False  # negligible + already tight


# --------------------------------------------------------------------------- #
# Calibration tiers
# --------------------------------------------------------------------------- #


def test_brier_score_ordering() -> None:
    assert brier_score([1.0, 0.0, 0.0], realized_index=0) == pytest.approx(0.0)
    confident_wrong = brier_score([0.0, 0.0, 1.0], realized_index=0)
    hedged = brier_score([1 / 3, 1 / 3, 1 / 3], realized_index=0)
    assert confident_wrong > hedged > 0.0


def test_tier2_tracker_brier_and_reliability() -> None:
    tracker = Tier2Tracker()
    rng = np.random.default_rng(0)
    # well-calibrated: realised category drawn from the predicted distribution
    for _ in range(400):
        p = rng.dirichlet([2, 2, 2])
        y = rng.choice(3, p=p)
        tracker.record(p, int(y))
    assert 0.0 < tracker.mean_brier() < 0.8
    curve = tracker.reliability_curve(n_bins=5)
    # predicted vs observed should track the diagonal for a calibrated forecaster
    for mean_pred, obs_freq, count in curve:
        if count >= 30:
            assert abs(mean_pred - obs_freq) < 0.15


def test_tier3_log_bayes_factor_rises_for_supporting_evidence() -> None:
    # evidence consistently more likely under the true regime
    traj = log_bayes_factor_trajectory(
        likelihoods_true=[0.6, 0.7, 0.65, 0.8],
        likelihoods_alt=[0.2, 0.3, 0.25, 0.2],
    )
    assert traj[-1] > traj[0]
    assert np.all(np.diff(traj) > 0)


def test_expert_weight_updates_reward_lower_brier() -> None:
    updates = {u.expert: u for u in update_weights({"good": 0.1, "bad": 1.2})}
    assert updates["good"].updated_weight > updates["bad"].updated_weight
    assert updates["good"].performance > updates["bad"].performance
    # a strong performer can earn a tight kappa cap; a weak one cannot
    assert updates["good"].kappa_cap == "tight"
    assert updates["bad"].kappa_cap in {"uncertain", "normal"}


def test_performance_from_brier_monotonic() -> None:
    assert performance_from_brier(0.0) == 1.0
    assert performance_from_brier(0.2) > performance_from_brier(0.8)


# --------------------------------------------------------------------------- #
# LLM-proposed CPTs
# --------------------------------------------------------------------------- #


class _FakeRetriever:
    def retrieve(self, query, k):
        return [AnalogEvent(text=f"analog {i}", citation=f"article:{i}#span") for i in range(k)]


def test_propose_cpt_returns_citations_and_commits_nothing() -> None:
    target = CPTColumnTarget(node="T", states=("none", "isolated", "frequent"), parent_config=("crisis",))
    client = ScriptedCompletionClient(seed_answers={}, target_answers={"T": (0.1, 0.3, 0.6)})
    proposal = propose_cpt(target, _FakeRetriever(), client, k=3)
    assert proposal.committed is False
    assert len(proposal.citations) == 3
    np.testing.assert_allclose(proposal.distribution, [0.1, 0.3, 0.6])
    target.validate_distribution(proposal.distribution)
