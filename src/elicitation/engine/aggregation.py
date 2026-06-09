"""Aggregation primitives: pooling several experts' distributions into one.

All three are pure functions over expert distributions (each a categorical
probability vector over the same ``K`` states). ``cooke_pool`` is the
performance-weighted linear pool that the Cooke protocol uses; the calibration
weights come from :mod:`src.elicitation.engine.calibration`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_distributions(distributions: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(distributions, dtype=float)
    if arr.ndim != 2:
        raise ValueError("distributions must be a 2-D array (n_experts, n_states)")
    if np.any(arr < 0):
        raise ValueError("distributions must be non-negative")
    row_sums = arr.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0):
        raise ValueError("each expert distribution must have positive mass")
    return arr / row_sums


def _resolve_weights(weights: ArrayLike | None, n: int) -> NDArray[np.float64]:
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"weights must have shape ({n},)")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = w.sum()
    if total == 0:
        # All experts zeroed (e.g. every expert failed Cooke calibration):
        # fall back to equal weight rather than divide by zero.
        return np.full(n, 1.0 / n)
    return w / total


def linear_pool(distributions: ArrayLike, weights: ArrayLike | None = None) -> NDArray[np.float64]:
    """Weighted arithmetic mean of expert distributions (renormalised)."""
    d = _as_distributions(distributions)
    w = _resolve_weights(weights, d.shape[0])
    pooled = (w[:, None] * d).sum(axis=0)
    return pooled / pooled.sum()


def logarithmic_pool(
    distributions: ArrayLike, weights: ArrayLike | None = None, eps: float = 1e-12
) -> NDArray[np.float64]:
    """Weighted geometric mean of expert distributions (renormalised)."""
    d = np.clip(_as_distributions(distributions), eps, None)
    w = _resolve_weights(weights, d.shape[0])
    log_pooled = (w[:, None] * np.log(d)).sum(axis=0)
    pooled = np.exp(log_pooled - log_pooled.max())
    return pooled / pooled.sum()


def cooke_pool(distributions: ArrayLike, cooke_weights: ArrayLike) -> NDArray[np.float64]:
    """Performance-weighted linear pool (Cooke's classical model aggregation).

    ``cooke_weights`` are the per-expert weights from calibration scoring; they
    may contain zeros (experts below the cutoff). If every weight is zero the
    pool falls back to equal weighting.
    """
    return linear_pool(distributions, cooke_weights)
