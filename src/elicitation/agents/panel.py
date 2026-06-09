"""Multi-model panel orchestration.

A :class:`Panel` runs Cooke over a set of (ideally multi-model) AI experts:
measures inter-agent correlation, feeds it into the kappa estimate (so a
correlated panel produces a wider, more honest interval), and assembles the
defensibility report. Roles compose with base-model diversity but are not
credited as independence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..engine.calibration import DEFAULT_QUANTILES
from ..engine.kappa import KappaLadder
from ..protocols.base import Expert, SeedQuestion
from ..protocols.cooke import CookeProtocol, CookeResult
from ..protocols.targets import CPTColumnTarget
from .calibration_report import panel_calibration_report
from .contamination import ProbeResult, summarize_probes
from .decorrelation import (
    CorrelationAdjustment,
    effective_sample_size,
    mean_pairwise_correlation,
)


def ensure_consistent_identity(scored: Expert, contributing: Expert) -> None:
    """Guard: a calibration score may only weight the *same* identity tuple.

    Prevents using a neutral-scored agent's weight for an in-role estimate
    (methodology §8.5) — the calibration unit is (base_model, role, config).
    """
    if scored.identity != contributing.identity:
        raise ValueError(
            f"scoring/identity mismatch: scored as {scored.identity}, "
            f"contributing as {contributing.identity}"
        )


@dataclass(frozen=True)
class PanelResult:
    cooke: CookeResult
    correlation: CorrelationAdjustment
    report: dict


class Panel:
    def __init__(self, experts: Sequence[Expert], ladder: KappaLadder | None = None) -> None:
        if len(experts) < 2:
            raise ValueError("a panel needs at least two experts")
        self.experts = list(experts)
        self.ladder = ladder or KappaLadder()

    def base_models(self) -> set[str]:
        return {e.base_model for e in self.experts if e.base_model}

    def is_multi_model(self) -> bool:
        return len(self.base_models()) >= 2

    def has_red_team(self) -> bool:
        return any((e.role or "").lower() in {"red-team", "red_team", "skeptic"} for e in self.experts)

    def measure_correlation(
        self, seeds: Sequence[SeedQuestion], quantile_levels: Sequence[float] = DEFAULT_QUANTILES
    ) -> CorrelationAdjustment:
        median_idx = len(quantile_levels) // 2
        series = [
            [e.answer_seed(s, quantile_levels).quantiles[median_idx] for s in seeds]
            for e in self.experts
        ]
        rho = mean_pairwise_correlation(series)
        n = len(self.experts)
        return CorrelationAdjustment(
            mean_correlation=rho,
            n_agents=n,
            effective_n=effective_sample_size(n, rho),
            distinct_base_models=len(self.base_models()),
        )

    def run_cooke(
        self,
        seeds: Sequence[SeedQuestion],
        targets: Sequence[CPTColumnTarget],
        alpha: float = 0.0,
        probe_results: list[ProbeResult] | None = None,
    ) -> PanelResult:
        adjustment = self.measure_correlation(seeds)
        cooke = CookeProtocol(self.ladder).run(
            self.experts, seeds, targets, alpha=alpha, correlation=adjustment.mean_correlation
        )
        report = panel_calibration_report(
            self.experts,
            cooke,
            adjustment,
            contamination_summary=summarize_probes(probe_results) if probe_results else None,
        )
        return PanelResult(cooke=cooke, correlation=adjustment, report=report)
