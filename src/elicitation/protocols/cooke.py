"""Cooke's classical model as a runnable protocol.

Seed scoring is performed once per panel (the expensive step) and the resulting
performance weights are reused across every CPT column the panel elicits. Each
target is aggregated by the performance-weighted linear pool; the per-CPT kappa
is estimated from panel spread, snapped to the ordinal ladder, and capped by the
panel's calibration. See ``docs/elicitation_methodology_and_defensibility.md``
§3.3 and §6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..engine.aggregation import cooke_pool
from ..engine.calibration import DEFAULT_QUANTILES, ExpertScore, classical_model_weights
from ..engine.kappa import KappaLadder, kappa_from_panel_spread
from .base import (
    Expert,
    ProvenanceMetadata,
    SeedQuestion,
    WorkflowSpec,
    WorkflowStep,
)
from .targets import CPTColumnTarget

WORKFLOW = WorkflowSpec(
    protocol="cooke",
    steps=(
        WorkflowStep("seed_elicitation", "Each expert answers the seed questions with quantiles."),
        WorkflowStep("scoring", "Score calibration x information; zero experts below the cutoff."),
        WorkflowStep("target_elicitation", "Each expert gives a distribution for each CPT column."),
        WorkflowStep("aggregation", "Performance-weighted pool; estimate, snap, and cap kappa."),
    ),
)


@dataclass(frozen=True)
class TargetResult:
    node: str
    parent_config: tuple[str, ...]
    mean: tuple[float, ...]
    kappa: float
    kappa_level: str
    kappa_raw: float
    provenance: ProvenanceMetadata


@dataclass(frozen=True)
class CookeResult:
    expert_scores: list[ExpertScore]
    weights: list[float]
    targets: dict[str, TargetResult]


class CookeProtocol:
    """The classical-model elicitation workflow."""

    def __init__(self, ladder: KappaLadder | None = None) -> None:
        self.ladder = ladder or KappaLadder()

    @staticmethod
    def workflow() -> WorkflowSpec:
        return WORKFLOW

    @staticmethod
    def required_experts() -> tuple[int, int]:
        return (4, 12)

    # -- scoring -------------------------------------------------------------

    def score_experts(
        self,
        experts: Sequence[Expert],
        seeds: Sequence[SeedQuestion],
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
        alpha: float = 0.0,
    ) -> list[ExpertScore]:
        experts_quantiles = [
            np.array([expert.answer_seed(s, quantile_levels).quantiles for s in seeds], dtype=float)
            for expert in experts
        ]
        realizations = np.array([s.realization for s in seeds], dtype=float)
        return classical_model_weights(experts_quantiles, realizations, quantile_levels, alpha)

    # -- aggregation ---------------------------------------------------------

    def aggregate_target(
        self,
        experts: Sequence[Expert],
        target: CPTColumnTarget,
        scores: Sequence[ExpertScore],
        correlation: float = 0.0,
    ) -> TargetResult:
        vectors = []
        for expert in experts:
            ans = expert.answer_target(target)
            target.validate_distribution(ans.probabilities)
            vectors.append(ans.probabilities)
        vectors = np.array(vectors, dtype=float)

        weights = np.array([s.weight for s in scores], dtype=float)
        mean = cooke_pool(vectors, weights)

        # kappa: estimate from the spread of the *contributing* experts only
        # (zeroed experts are excluded from the pool, so they should not inflate
        # the spread either), snap to the ladder, then cap by calibration.
        contributing = vectors[weights > 0]
        if len(contributing) >= 2:
            kappa_raw = kappa_from_panel_spread(contributing, correlation=correlation)
        else:
            kappa_raw = self.ladder.kappa_for("uncertain")
        level = self.ladder.snap(kappa_raw)
        cal_bar = float(np.sum(weights * np.array([s.calibration for s in scores])))
        capped = self.ladder.cap(level, cal_bar)
        kappa = self.ladder.kappa_for(capped)

        is_ai = any(e.kind == "ai" for e in experts)
        model_set = {
            "models": sorted({e.base_model for e in experts if e.base_model}),
            "n_experts": len(experts),
            "roles": sorted({e.role for e in experts if e.role}),
        }
        provenance = ProvenanceMetadata(
            protocol="cooke",
            kappa=kappa,
            kappa_level=capped,
            calibration_score=cal_bar,
            model_set=model_set,
            weights={e.name: float(w) for e, w in zip(experts, weights)},
            is_ai_sourced=is_ai,
            correlation_note=(f"correlation={correlation:.2f}" if correlation else None),
        )
        return TargetResult(
            node=target.node,
            parent_config=target.parent_config,
            mean=tuple(float(x) for x in mean),
            kappa=kappa,
            kappa_level=capped,
            kappa_raw=float(kappa_raw),
            provenance=provenance,
        )

    # -- end-to-end ----------------------------------------------------------

    def run(
        self,
        experts: Sequence[Expert],
        seeds: Sequence[SeedQuestion],
        targets: Sequence[CPTColumnTarget],
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
        alpha: float = 0.0,
        correlation: float = 0.0,
    ) -> CookeResult:
        scores = self.score_experts(experts, seeds, quantile_levels, alpha)
        results = {
            t.node: self.aggregate_target(experts, t, scores, correlation) for t in targets
        }
        return CookeResult(
            expert_scores=scores,
            weights=[s.weight for s in scores],
            targets=results,
        )
