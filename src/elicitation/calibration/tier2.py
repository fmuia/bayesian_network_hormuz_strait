"""Tier 2 calibration: intermediate-node outcome tracking.

For nodes whose outcomes can be observed, record the realised state and score the
model's prediction. Brier scores and reliability curves accumulate over time and
become the empirical component of the confidence statement — the only
out-of-sample evidence, and the only thing that can catch a wrong structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike


def brier_score(predicted: ArrayLike, realized_index: int) -> float:
    """Multi-category Brier score: sum_i (p_i - y_i)^2, y one-hot. 0 is perfect."""
    p = np.asarray(predicted, dtype=float)
    y = np.zeros_like(p)
    y[realized_index] = 1.0
    return float(np.sum((p - y) ** 2))


@dataclass
class Tier2Tracker:
    """Accumulates (predicted distribution, realised state) observations."""

    records: list[tuple[np.ndarray, int]] = field(default_factory=list)

    def record(self, predicted: ArrayLike, realized_index: int) -> None:
        self.records.append((np.asarray(predicted, dtype=float), realized_index))

    def mean_brier(self) -> float:
        if not self.records:
            raise ValueError("no records")
        return float(np.mean([brier_score(p, y) for p, y in self.records]))

    def reliability_curve(self, n_bins: int = 10) -> list[tuple[float, float, int]]:
        """Reliability curve over the predicted probability of the realised-vs-not
        event, as ``(mean_predicted, observed_frequency, count)`` per bin.

        Uses the predicted probability assigned to each realised category."""
        preds = np.array([p[y] for p, y in self.records])
        hits = np.ones(len(preds))  # the realised category did occur
        # also include the non-realised categories as negatives for a fair curve
        neg_preds, neg_hits = [], []
        for p, y in self.records:
            for i in range(len(p)):
                if i != y:
                    neg_preds.append(p[i])
                    neg_hits.append(0.0)
        all_preds = np.concatenate([preds, np.array(neg_preds)]) if neg_preds else preds
        all_hits = np.concatenate([hits, np.array(neg_hits)]) if neg_hits else hits

        edges = np.linspace(0, 1, n_bins + 1)
        curve = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (all_preds >= lo) & (all_preds < hi if hi < 1 else all_preds <= hi)
            if mask.sum() == 0:
                continue
            curve.append(
                (float(all_preds[mask].mean()), float(all_hits[mask].mean()), int(mask.sum()))
            )
        return curve


__all__ = ["brier_score", "Tier2Tracker"]
