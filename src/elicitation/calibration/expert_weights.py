"""Update expert weights and kappa caps from accrued outcome calibration.

As an expert's contributions are scored against realised outcomes (Tier 2), its
weight in future elicitations and the kappa level it may contribute are updated.
Better realised performance (lower Brier) raises the weight; sustained good
performance lifts the kappa cap. This feeds back into Cooke (Layer 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..engine.kappa import KappaLadder


@dataclass(frozen=True)
class WeightUpdate:
    expert: str
    performance: float       # in (0, 1]; higher is better
    updated_weight: float
    kappa_cap: str


def performance_from_brier(mean_brier: float, max_brier: float = 2.0) -> float:
    """Map a mean Brier score to a performance score in (0, 1] (1 = perfect)."""
    return float(np.clip(1.0 - mean_brier / max_brier, 0.0, 1.0))


def update_weights(
    brier_by_expert: dict[str, float],
    ladder: KappaLadder | None = None,
) -> list[WeightUpdate]:
    """Recompute normalised weights and kappa caps from realised Brier scores."""
    ladder = ladder or KappaLadder()
    perf = {e: performance_from_brier(b) for e, b in brier_by_expert.items()}
    total = sum(perf.values())
    updates = []
    for expert, p in perf.items():
        weight = p / total if total > 0 else 1.0 / len(perf)
        # reuse the calibration->level cap: performance acts like a calibration score
        cap = ladder.cap("tight", calibration_score=p)
        updates.append(WeightUpdate(expert=expert, performance=p, updated_weight=weight, kappa_cap=cap))
    return updates


__all__ = ["WeightUpdate", "performance_from_brier", "update_weights"]
