"""Tests for the Layer 1 core engine."""

from __future__ import annotations

import numpy as np
import pytest

from src.elicitation.engine import (
    KappaLadder,
    calibration_score,
    classical_model_weights,
    cooke_pool,
    information_score,
    kappa_from_panel_spread,
    kappa_from_seed_coverage,
    linear_pool,
    logarithmic_pool,
    morris_screening,
    posterior_variance_decomposition,
    sobol_indices,
)

# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def test_pools_are_identity_for_single_expert() -> None:
    d = [[0.2, 0.3, 0.5]]
    np.testing.assert_allclose(linear_pool(d), d[0])
    np.testing.assert_allclose(logarithmic_pool(d), d[0], atol=1e-9)


def test_linear_pool_two_experts_symmetric() -> None:
    pooled = linear_pool([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(pooled, [0.5, 0.5])


def test_linear_pool_weighted() -> None:
    pooled = linear_pool([[1.0, 0.0], [0.0, 1.0]], weights=[3.0, 1.0])
    np.testing.assert_allclose(pooled, [0.75, 0.25])


def test_logarithmic_pool_is_more_concentrated_than_linear() -> None:
    d = [[0.6, 0.4], [0.7, 0.3]]
    lin = linear_pool(d)
    log = logarithmic_pool(d)
    # geometric pooling concentrates toward the agreed-upon larger mass
    assert log[0] >= lin[0]


def test_cooke_pool_falls_back_to_equal_when_all_zero() -> None:
    pooled = cooke_pool([[1.0, 0.0], [0.0, 1.0]], cooke_weights=[0.0, 0.0])
    np.testing.assert_allclose(pooled, [0.5, 0.5])


def test_pool_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        linear_pool([[-0.1, 1.1]])


# --------------------------------------------------------------------------- #
# Cooke classical-model scoring
# --------------------------------------------------------------------------- #


def _fixed_quantiles(mu: float, sigma: float, n_seeds: int) -> np.ndarray:
    """A well-calibrated expert states the true (5, 50, 95) quantiles of the
    generating distribution for every seed."""
    q = np.array([mu - 1.645 * sigma, mu, mu + 1.645 * sigma])
    return np.tile(q, (n_seeds, 1))


def test_well_calibrated_expert_scores_higher_than_overconfident() -> None:
    rng = np.random.default_rng(0)
    mu, sigma, n = 10.0, 3.0, 200
    realizations = rng.normal(mu, sigma, size=n)
    # genuinely calibrated: stated quantiles match the generating distribution
    good = _fixed_quantiles(mu, sigma, n)
    # overconfident: a narrow band that rarely contains the truth
    overconfident = np.tile([mu - 0.2, mu, mu + 0.2], (n, 1))
    c_good = calibration_score(good, realizations)
    c_bad = calibration_score(overconfident, realizations)
    assert c_good > c_bad
    assert c_good > 0.5 and c_bad < 0.05


def test_information_rewards_tighter_distributions() -> None:
    realizations = np.linspace(0, 10, 20)
    rng = (-5.0, 15.0)
    tight = np.column_stack([realizations - 0.5, realizations, realizations + 0.5])
    vague = np.column_stack([realizations - 6.0, realizations, realizations + 6.0])
    assert information_score(tight, rng=rng) > information_score(vague, rng=rng)


def test_classical_model_zeroes_expert_below_cutoff() -> None:
    rng = np.random.default_rng(1)
    mu, sigma, n = 0.0, 1.0, 200
    realizations = rng.normal(mu, sigma, size=n)
    good = _fixed_quantiles(mu, sigma, n)                       # well-calibrated
    bad = np.tile([mu + 3, mu + 4, mu + 5], (n, 1))            # confidently wrong
    scores = classical_model_weights([good, bad], realizations, alpha=0.05)
    assert scores[1].weight == 0.0                              # bad expert zeroed
    assert scores[0].weight == pytest.approx(1.0)              # good expert carries all weight


# --------------------------------------------------------------------------- #
# Kappa mapping
# --------------------------------------------------------------------------- #


def test_kappa_decreases_monotonically_with_correlation() -> None:
    vectors = np.array([[0.1, 0.3, 0.6], [0.12, 0.34, 0.54], [0.08, 0.28, 0.64], [0.1, 0.32, 0.58]])
    kappas = [kappa_from_panel_spread(vectors, correlation=r) for r in (0.0, 0.3, 0.6, 0.9)]
    assert all(earlier >= later for earlier, later in zip(kappas, kappas[1:]))
    assert kappas[0] > kappas[-1]


def test_kappa_from_seed_coverage_recovers_known_concentration() -> None:
    rng = np.random.default_rng(7)
    mean = np.array([0.2, 0.3, 0.5])
    kappa_true = 25.0
    n_seeds, draws = 400, 50
    means = np.tile(mean, (n_seeds, 1))
    counts = np.array(
        [rng.multinomial(draws, rng.dirichlet(kappa_true * mean)) for _ in range(n_seeds)]
    )
    estimated = kappa_from_seed_coverage(means, counts)
    assert estimated == pytest.approx(kappa_true, rel=0.35)


def test_kappa_ladder_snap_and_cap() -> None:
    ladder = KappaLadder()
    assert ladder.snap(40) == "tight"
    assert ladder.snap(15) == "normal"
    assert ladder.snap(5) == "uncertain"
    # a poorly-calibrated expert cannot claim "tight"
    assert ladder.cap("tight", calibration_score=0.1) == "uncertain"
    assert ladder.cap("tight", calibration_score=0.7) == "tight"
    # capping never raises a level
    assert ladder.cap("uncertain", calibration_score=0.9) == "uncertain"


# --------------------------------------------------------------------------- #
# Sensitivity
# --------------------------------------------------------------------------- #


def test_sobol_matches_ishigami_reference() -> None:
    a, b = 7.0, 0.1

    def ishigami(x: np.ndarray) -> np.ndarray:
        return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])

    bounds = [(-np.pi, np.pi)] * 3
    res = sobol_indices(ishigami, bounds, n=40000, seed=0)
    # analytic first-order indices ~ (0.314, 0.442, 0.0)
    assert res.first_order[0] == pytest.approx(0.314, abs=0.06)
    assert res.first_order[1] == pytest.approx(0.442, abs=0.06)
    assert abs(res.first_order[2]) < 0.05
    # x3 has no first-order effect but a non-trivial total (interaction with x1)
    assert res.total[2] > 0.1


def test_morris_flags_influential_inputs_and_ignores_dummy() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        return 5.0 * x[:, 0] + 0.5 * x[:, 1] + 0.0 * x[:, 2]

    res = morris_screening(f, [(0, 1)] * 3, trajectories=300, seed=0)
    assert res.mu_star[0] > res.mu_star[1] > res.mu_star[2]
    assert res.mu_star[2] == pytest.approx(0.0, abs=1e-9)


def test_posterior_variance_decomposition_attributes_to_dominant_cpt() -> None:
    # output depends almost entirely on CPT "A"; "B" is nearly fixed (high kappa)
    cpts = {"A": ([0.5, 0.5], 6.0), "B": ([0.5, 0.5], 500.0)}

    def output(sample: dict[str, np.ndarray]) -> float:
        return float(sample["A"][0] + 0.01 * sample["B"][0])

    decomp = posterior_variance_decomposition(cpts, output, n_samples=2000, seed=0)
    assert decomp.shares["A"] > 0.95
    assert decomp.total_variance > 0
