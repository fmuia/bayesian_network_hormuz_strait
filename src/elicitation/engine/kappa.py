"""The calibration -> kappa mapping.

A CPT column is a categorical distribution; we represent uncertainty about it
with a Dirichlet whose mean is the elicited vector and whose scalar
concentration ``kappa`` is set from measured quality. Two routes (methodology
§6):

* :func:`kappa_from_panel_spread` — method-of-moments from how much a panel
  disagrees, discounted by measured inter-agent correlation (effective N).
* :func:`kappa_from_seed_coverage` — concentration that best fits observed seed
  outcomes (Dirichlet-multinomial MLE), the realisation of coverage calibration.

The continuous estimate is then snapped to a three-level ordinal ladder
(``tight`` / ``normal`` / ``uncertain``) for reporting, and an expert's measured
calibration caps the level it may contribute.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

KAPPA_MIN = 1.0
KAPPA_MAX = 1000.0


def dirichlet_variance(mean: ArrayLike, kappa: float) -> NDArray[np.float64]:
    """Per-component variance of Dirichlet(kappa * mean): m_i(1-m_i)/(kappa+1)."""
    m = np.asarray(mean, dtype=float)
    return m * (1.0 - m) / (kappa + 1.0)


def _effective_n(n: int, correlation: float) -> float:
    """Effective independent sample size for ``n`` agents with mean pairwise
    correlation ``correlation`` (in [0, 1]). n_eff = n / (1 + (n-1) rho)."""
    rho = float(np.clip(correlation, 0.0, 1.0))
    return n / (1.0 + (n - 1) * rho)


def kappa_from_panel_spread(
    expert_vectors: ArrayLike,
    correlation: float = 0.0,
    kappa_bounds: tuple[float, float] = (KAPPA_MIN, KAPPA_MAX),
) -> float:
    """Estimate kappa from the dispersion of a panel's point vectors.

    For each state, invert the Dirichlet variance: kappa_i = m_i(1-m_i)/s_i^2 - 1.
    Pool the per-state estimates (median of the finite, positive ones), then
    apply a correlation discount so that more correlated panels (less
    independent information) yield a *lower* kappa — i.e. a wider, more honest
    credible interval.
    """
    v = np.asarray(expert_vectors, dtype=float)
    if v.ndim != 2 or v.shape[0] < 2:
        raise ValueError("expert_vectors must be (n_experts >= 2, n_states)")
    n = v.shape[0]
    mean = v.mean(axis=0)
    var = v.var(axis=0, ddof=1)

    estimates = []
    for m_i, s2_i in zip(mean, var):
        if 0.0 < m_i < 1.0 and s2_i > 0:
            estimates.append(m_i * (1.0 - m_i) / s2_i - 1.0)
    raw = float(np.median(estimates)) if estimates else kappa_bounds[1]

    discount = _effective_n(n, correlation) / n  # in (0, 1], = 1 when rho = 0
    kappa = raw * discount
    return float(np.clip(kappa, *kappa_bounds))


def kappa_from_seed_coverage(
    means: ArrayLike,
    count_observations: ArrayLike,
    kappa_bounds: tuple[float, float] = (KAPPA_MIN, KAPPA_MAX),
) -> float:
    """Concentration that best fits observed seed outcomes.

    ``means`` (n_seeds, n_states) are the predicted mean distributions; each row
    of ``count_observations`` (n_seeds, n_states) is the realised category counts
    for that seed. Returns the kappa maximising the Dirichlet-multinomial
    likelihood — the value at which the model's stated uncertainty matches the
    realised spread (coverage calibration). Categorical coverage-fit via a
    proper scoring rule is the open research refinement (methodology §6.1).
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import gammaln

    m = np.asarray(means, dtype=float)
    c = np.asarray(count_observations, dtype=float)
    if m.shape != c.shape or m.ndim != 2:
        raise ValueError("means and count_observations must share shape (n_seeds, n_states)")

    def neg_log_likelihood(log_kappa: float) -> float:
        kappa = np.exp(log_kappa)
        alpha = kappa * m
        n = c.sum(axis=1)
        # Dirichlet-multinomial log-pmf per seed (drop the constant binomial term).
        ll = (
            gammaln(alpha.sum(axis=1))
            - gammaln(alpha.sum(axis=1) + n)
            + np.sum(gammaln(alpha + c) - gammaln(alpha), axis=1)
        )
        return -float(np.sum(ll))

    lo, hi = np.log(kappa_bounds[0]), np.log(kappa_bounds[1])
    result = minimize_scalar(neg_log_likelihood, bounds=(lo, hi), method="bounded")
    return float(np.clip(np.exp(result.x), *kappa_bounds))


@dataclass(frozen=True)
class KappaLadder:
    """The three-level ordinal kappa ladder, with calibration-based caps.

    ``levels`` maps a level name to its kappa value (per-deployment fitted).
    ``caps`` maps a level to the minimum calibration score required to claim it.
    Levels are ordered uncertain < normal < tight by their kappa value.
    """

    levels: dict[str, float] = field(
        default_factory=lambda: {"uncertain": 5.0, "normal": 15.0, "tight": 40.0}
    )
    caps: dict[str, float] = field(
        default_factory=lambda: {"uncertain": 0.0, "normal": 0.3, "tight": 0.6}
    )

    def _ordered(self) -> list[str]:
        return sorted(self.levels, key=lambda lvl: self.levels[lvl])

    def snap(self, kappa: float) -> str:
        """Nearest level to ``kappa`` (compared in log space)."""
        log_k = np.log(kappa)
        return min(self.levels, key=lambda lvl: abs(np.log(self.levels[lvl]) - log_k))

    def kappa_for(self, level: str) -> float:
        return self.levels[level]

    def cap(self, level: str, calibration_score: float) -> str:
        """Lower ``level`` to the highest the calibration score permits."""
        ordered = self._ordered()
        allowed = ordered[0]
        for lvl in ordered:
            if calibration_score >= self.caps[lvl]:
                allowed = lvl
        # take the lower of (proposed level, allowed level)
        if ordered.index(level) <= ordered.index(allowed):
            return level
        return allowed


def snap_to_level(kappa: float, ladder: KappaLadder | None = None) -> str:
    """Snap a continuous kappa to the nearest ordinal level."""
    return (ladder or KappaLadder()).snap(kappa)


def cap_level(level: str, calibration_score: float, ladder: KappaLadder | None = None) -> str:
    """Cap a proposed level by an expert's measured calibration."""
    return (ladder or KappaLadder()).cap(level, calibration_score)
