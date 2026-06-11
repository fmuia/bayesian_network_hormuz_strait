"""Cooke's classical model: scoring experts on seed (calibration) questions.

An expert answers each continuous seed question with quantiles (e.g. the 5th,
50th and 95th percentiles). Those quantiles partition the range into bins with
known theoretical probabilities. Two scores follow:

* **Calibration (statistical accuracy).** How well the realised seed values fall
  into the bins at the asserted rates. The test statistic ``2 N I(s, p)`` —
  where ``s`` is the empirical bin distribution, ``p`` the theoretical one, ``I``
  the Kullback-Leibler divergence, and ``N`` the number of seeds — is
  asymptotically chi-square with ``(bins - 1)`` degrees of freedom. The
  calibration score is its p-value: high means well-calibrated.
* **Information.** How tight the expert's distributions are relative to a
  background measure over the intrinsic range. Higher means more informative.

The combined (unnormalised) weight is ``calibration * information``, zeroed for
experts whose calibration falls below the cutoff ``alpha``. References:
Cooke (1991); Cooke & Goossens (2008). See
``docs/elicitation_methodology_and_defensibility.md`` §3.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.50, 0.95)


def bin_probabilities(quantile_levels: ArrayLike = DEFAULT_QUANTILES) -> NDArray[np.float64]:
    """Theoretical probabilities of the inter-quantile bins.

    For levels (0.05, 0.5, 0.95) the bins are (.., 5%], (5,50], (50,95], (95,..)
    with probabilities (0.05, 0.45, 0.45, 0.05).
    """
    q = np.asarray(quantile_levels, dtype=float)
    if np.any(np.diff(q) <= 0) or q[0] <= 0 or q[-1] >= 1:
        raise ValueError("quantile_levels must be strictly increasing within (0, 1)")
    edges = np.concatenate([[0.0], q, [1.0]])
    return np.diff(edges)


def intrinsic_range(
    expert_quantiles: ArrayLike, realizations: ArrayLike, overshoot: float = 0.1
) -> tuple[float, float]:
    """The [low, high] range used for information scoring, with k% overshoot.

    Spans every elicited quantile and every realised value, widened by
    ``overshoot`` of that span at each end (Cooke's default is 10%).
    """
    qs = np.asarray(expert_quantiles, dtype=float)
    r = np.asarray(realizations, dtype=float)
    lo = float(min(qs.min(), r.min()))
    hi = float(max(qs.max(), r.max()))
    span = hi - lo
    if span == 0:
        span = 1.0
    return lo - overshoot * span, hi + overshoot * span


def _bin_index(realization: float, quantile_values: NDArray[np.float64]) -> int:
    """Which bin a realised value falls in (0 .. n_quantiles)."""
    return int(np.searchsorted(quantile_values, realization, side="right"))


def calibration_score(
    expert_quantiles: ArrayLike,
    realizations: ArrayLike,
    quantile_levels: ArrayLike = DEFAULT_QUANTILES,
) -> float:
    """Statistical-accuracy p-value for one expert over the seed set.

    ``expert_quantiles`` has shape (n_seeds, n_levels); ``realizations`` has
    shape (n_seeds,). Returns a value in [0, 1] — higher is better calibrated.
    """
    q = np.asarray(expert_quantiles, dtype=float)
    r = np.asarray(realizations, dtype=float)
    if q.ndim != 2 or q.shape[0] != r.shape[0]:
        raise ValueError("expert_quantiles must be (n_seeds, n_levels) aligned with realizations")
    n_seeds = q.shape[0]
    p = bin_probabilities(quantile_levels)
    n_bins = p.size

    counts = np.zeros(n_bins)
    for i in range(n_seeds):
        counts[_bin_index(float(r[i]), np.sort(q[i]))] += 1
    s = counts / n_seeds

    # KL divergence I(s || p); 0 * log(0) := 0.
    mask = s > 0
    kl = float(np.sum(s[mask] * np.log(s[mask] / p[mask])))
    statistic = 2.0 * n_seeds * kl
    return float(chi2.sf(statistic, df=n_bins - 1))


def information_score(
    expert_quantiles: ArrayLike,
    quantile_levels: ArrayLike = DEFAULT_QUANTILES,
    rng: tuple[float, float] | None = None,
    realizations: ArrayLike | None = None,
) -> float:
    """Mean relative information of the expert's distributions vs a uniform
    background over the intrinsic range. Higher means tighter / more decisive.
    """
    q = np.asarray(expert_quantiles, dtype=float)
    p = bin_probabilities(quantile_levels)
    if rng is None:
        if realizations is None:
            raise ValueError("provide either rng or realizations to set the intrinsic range")
        rng = intrinsic_range(q, realizations)
    low, high = rng
    width = high - low
    if width <= 0:
        raise ValueError("intrinsic range must have positive width")

    infos = []
    for i in range(q.shape[0]):
        edges = np.concatenate([[low], np.sort(q[i]), [high]])
        bin_widths = np.diff(edges)
        if np.any(bin_widths <= 0):
            # Quantiles outside the intrinsic range; clamp to keep widths positive.
            edges = np.clip(edges, low, high)
            edges = np.maximum.accumulate(edges)
            bin_widths = np.clip(np.diff(edges), 1e-12, None)
        background = bin_widths / width
        infos.append(float(np.sum(p * np.log(p / background))))
    return float(np.mean(infos))


@dataclass(frozen=True)
class ExpertScore:
    """Calibration, information, and the resulting (normalised) Cooke weight."""

    calibration: float
    information: float
    raw_weight: float
    weight: float


def classical_model_weights(
    experts_quantiles: list[ArrayLike],
    realizations: ArrayLike,
    quantile_levels: ArrayLike = DEFAULT_QUANTILES,
    alpha: float = 0.0,
) -> list[ExpertScore]:
    """Score every expert and return normalised performance weights.

    ``alpha`` is the calibration cutoff: experts with ``calibration < alpha`` are
    zeroed (Cooke's poorly-calibrated-experts-discarded rule). With ``alpha = 0``
    no expert is zeroed. A shared intrinsic range across all experts is used for
    the information score so the comparison is fair.
    """
    r = np.asarray(realizations, dtype=float)
    all_q = np.concatenate([np.asarray(q, dtype=float).ravel() for q in experts_quantiles])
    span_lo, span_hi = intrinsic_range(all_q, r)

    calibrations, informations = [], []
    for q in experts_quantiles:
        calibrations.append(calibration_score(q, r, quantile_levels))
        informations.append(information_score(q, quantile_levels, rng=(span_lo, span_hi)))

    raw = [
        (c * i if c >= alpha else 0.0)
        for c, i in zip(calibrations, informations)
    ]
    total = sum(raw)
    weights = [w / total for w in raw] if total > 0 else [1.0 / len(raw)] * len(raw)

    return [
        ExpertScore(calibration=c, information=i, raw_weight=rw, weight=w)
        for c, i, rw, w in zip(calibrations, informations, raw, weights)
    ]
