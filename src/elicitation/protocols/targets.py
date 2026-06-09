"""Elicitation targets. The single in-scope shape is a CPT column."""

from __future__ import annotations

from dataclasses import dataclass

from .base import ElicitationTarget


@dataclass(frozen=True)
class CPTColumnTarget(ElicitationTarget):
    """One column of a CPT: ``P(node | parents = parent_config)``.

    The elicited quantity is a categorical distribution over ``states``.
    """

    states: tuple[str, ...] = ()
    parent_config: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.states) < 2:
            raise ValueError("a CPT column must have at least two states")

    def n_outcomes(self) -> int:
        return len(self.states)

    def describe(self) -> str:
        parents = ", ".join(self.parent_config) if self.parent_config else "(root)"
        return (
            f"P({self.node} | {parents}) over states "
            f"[{', '.join(self.states)}]"
        )

    def validate_distribution(self, probabilities) -> None:
        if len(probabilities) != len(self.states):
            raise ValueError(
                f"distribution length {len(probabilities)} != {len(self.states)} states"
            )
        total = sum(probabilities)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"distribution must sum to 1, got {total}")
