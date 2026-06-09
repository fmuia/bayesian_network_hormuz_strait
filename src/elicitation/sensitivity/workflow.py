"""Sensitivity-driven prioritisation workflow (Plan 4, Layer 6.2).

Wraps the engine's variance decomposition in an analyst-facing view: which CPTs
dominate the output's uncertainty, and where tightening kappa would most reduce
the credible interval. This directs re-elicitation effort; it is not a gate on
whether elicitation happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from numpy.typing import NDArray

from ..engine.sensitivity import VarianceDecomposition, posterior_variance_decomposition


@dataclass(frozen=True)
class Priority:
    node: str
    variance_share: float
    kappa_level: str | None
    recommended: bool


def prioritise(
    decomposition: VarianceDecomposition,
    kappa_levels: Mapping[str, str] | None = None,
    tighten_levels: Sequence[str] = ("uncertain", "normal"),
) -> list[Priority]:
    """Rank CPTs by their share of output variance.

    A CPT is *recommended* for tightening when it carries a meaningful variance
    share and its current kappa level is still loose (in ``tighten_levels``).
    """
    levels = kappa_levels or {}
    ranked = sorted(decomposition.shares.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for node, share in ranked:
        level = levels.get(node)
        recommended = share >= 0.1 and (level is None or level in tighten_levels)
        out.append(Priority(node=node, variance_share=share, kappa_level=level, recommended=recommended))
    return out


def run_prioritisation(
    cpts: Mapping[str, tuple[Sequence[float], float]],
    output_fn: Callable[[Mapping[str, NDArray], None], float] | Callable,
    kappa_levels: Mapping[str, str] | None = None,
    n_samples: int = 512,
    seed: int | None = 0,
) -> list[Priority]:
    """Decompose output variance over the CPTs and rank them."""
    decomposition = posterior_variance_decomposition(cpts, output_fn, n_samples=n_samples, seed=seed)
    return prioritise(decomposition, kappa_levels)


__all__ = ["Priority", "prioritise", "run_prioritisation"]
