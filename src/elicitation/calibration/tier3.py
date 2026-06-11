"""Tier 3 calibration: Bayes-factor regime trajectory.

For the latent-regime model, record the cumulative log-Bayes-factor for the
expert-judged "true" regime against an alternative as evidence accrues over a
historical analog. A systematically declining trajectory reveals CPT regions
that are miscalibrated for that regime.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def log_bayes_factor_trajectory(
    likelihoods_true: ArrayLike,
    likelihoods_alt: ArrayLike,
    prior_true: float = 0.5,
    prior_alt: float = 0.5,
) -> np.ndarray:
    """Cumulative log Bayes factor for the true regime vs an alternative.

    ``likelihoods_true``/``likelihoods_alt`` are per-evidence-increment
    likelihoods P(evidence_t | regime). Returns the cumulative log BF after each
    increment; an increasing trajectory means the evidence supports the true
    regime (well-calibrated emissions).
    """
    lt = np.asarray(likelihoods_true, dtype=float)
    la = np.asarray(likelihoods_alt, dtype=float)
    if lt.shape != la.shape:
        raise ValueError("likelihood arrays must align")
    if np.any(lt <= 0) or np.any(la <= 0):
        raise ValueError("likelihoods must be positive")
    prior_term = np.log(prior_true / prior_alt)
    return prior_term + np.cumsum(np.log(lt) - np.log(la))


def final_log_bayes_factor(*args, **kwargs) -> float:
    return float(log_bayes_factor_trajectory(*args, **kwargs)[-1])


__all__ = ["log_bayes_factor_trajectory", "final_log_bayes_factor"]
