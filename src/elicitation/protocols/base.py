"""Protocol-agnostic interfaces shared by all elicitation protocols.

Only Cooke is implemented (Plan 4, decision B.6), but the abstractions here are
protocol- and expert-agnostic so IDEA/SHELF and human experts slot in later
without restructuring. An ``Expert`` is anything that can answer a seed question
with quantiles and a target with a categorical distribution — a human, or an
``LLMExpert`` (Layer 2/3). An AI expert's identity is the tuple
``(base_model, role, config)``; calibration is measured per tuple.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence


def config_fingerprint(config: Mapping | None) -> str:
    """A short, stable fingerprint of an expert's config (for identity)."""
    if not config:
        return "none"
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Questions and answers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SeedQuestion:
    """A continuous calibration question with a known answer (the realization).

    The realization is *not* shown to the expert; it is used only for scoring.
    """

    id: str
    text: str
    realization: float
    unit: str | None = None


@dataclass(frozen=True)
class QuantileAnswer:
    seed_id: str
    quantiles: tuple[float, ...]


@dataclass(frozen=True)
class DistributionAnswer:
    node: str
    parent_config: tuple[str, ...]
    probabilities: tuple[float, ...]


# --------------------------------------------------------------------------- #
# Elicitation targets
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ElicitationTarget(ABC):
    """Base class for what a protocol elicits. The only in-scope shape is a CPT
    column (see :class:`~src.elicitation.protocols.targets.CPTColumnTarget`)."""

    node: str

    @abstractmethod
    def n_outcomes(self) -> int: ...

    @abstractmethod
    def describe(self) -> str: ...


# --------------------------------------------------------------------------- #
# Experts
# --------------------------------------------------------------------------- #


class Expert(ABC):
    """A panel member — human or AI — driven through a protocol's steps."""

    def __init__(
        self,
        name: str,
        kind: str = "human",
        base_model: str | None = None,
        role: str | None = None,
        config: Mapping | None = None,
    ) -> None:
        if kind not in ("human", "ai"):
            raise ValueError("kind must be 'human' or 'ai'")
        self.name = name
        self.kind = kind
        self.base_model = base_model
        self.role = role
        self.config = dict(config) if config else None
        self.config_fingerprint = config_fingerprint(self.config)

    @property
    def identity(self) -> tuple[str | None, str | None, str]:
        """The (base_model, role, config) tuple calibration is measured against."""
        return (self.base_model, self.role, self.config_fingerprint)

    @abstractmethod
    def answer_seed(
        self, question: SeedQuestion, quantile_levels: Sequence[float]
    ) -> QuantileAnswer: ...

    @abstractmethod
    def answer_target(self, target: ElicitationTarget) -> DistributionAnswer: ...


class ScriptedExpert(Expert):
    """An expert backed by pre-supplied answers. Models a human's recorded
    judgments and is the deterministic stand-in for protocol tests."""

    def __init__(
        self,
        name: str,
        seed_answers: Mapping[str, Sequence[float]],
        target_answers: Mapping[str, Sequence[float]],
        **identity,
    ) -> None:
        super().__init__(name, **identity)
        self._seed = {k: tuple(v) for k, v in seed_answers.items()}
        self._target = {k: tuple(v) for k, v in target_answers.items()}

    def answer_seed(self, question: SeedQuestion, quantile_levels: Sequence[float]) -> QuantileAnswer:
        return QuantileAnswer(seed_id=question.id, quantiles=self._seed[question.id])

    def answer_target(self, target: ElicitationTarget) -> DistributionAnswer:
        probs = self._target[target.node]
        return DistributionAnswer(node=target.node, parent_config=getattr(target, "parent_config", ()), probabilities=probs)


# --------------------------------------------------------------------------- #
# Workflow + provenance
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WorkflowStep:
    name: str
    description: str


@dataclass(frozen=True)
class WorkflowSpec:
    protocol: str
    steps: tuple[WorkflowStep, ...]


@dataclass(frozen=True)
class ProvenanceMetadata:
    """What gets written to ``cpt_provenance`` at the conclusion of a run."""

    protocol: str
    kappa: float
    kappa_level: str
    calibration_score: float
    model_set: dict
    weights: dict
    is_ai_sourced: bool
    correlation_note: str | None = None
    contamination_summary: dict | None = None
    references: dict | None = None
