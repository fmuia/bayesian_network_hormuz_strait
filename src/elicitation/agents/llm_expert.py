"""LLM experts.

An :class:`LLMExpert` implements the *same* :class:`Expert` interface a human
uses, delegating the actual model call to a :class:`CompletionClient`. The
client is the only place a provider key is used (revealed at that boundary
only). Tests use the deterministic :class:`ScriptedCompletionClient`; the real
:class:`OpenAICompletionClient` is a thin adapter whose parsing is unit-tested
without any network call.

Identity is the tuple ``(base_model, role, config)`` (decision B.16): roles are
part of identity, calibration is measured per tuple, and a role used to set
scored estimates must be seed-scored in that same role.
"""

from __future__ import annotations

import json
import re
from typing import Protocol, Sequence

import numpy as np

from ..protocols.base import (
    DistributionAnswer,
    ElicitationTarget,
    Expert,
    QuantileAnswer,
    SeedQuestion,
)


def normalize(probabilities: Sequence[float]) -> tuple[float, ...]:
    """Clip negatives and renormalise to a valid probability vector."""
    arr = np.clip(np.asarray(probabilities, dtype=float), 0.0, None)
    total = arr.sum()
    if total <= 0:
        raise ValueError("model returned an all-zero distribution")
    return tuple(float(x) for x in arr / total)


class CompletionClient(Protocol):
    """The LLM boundary. Implementations turn a prompt into structured output."""

    def seed_quantiles(
        self, question_text: str, quantile_levels: Sequence[float], role: str | None
    ) -> list[float]: ...

    def target_distribution(
        self, node: str, prompt: str, states: Sequence[str], role: str | None
    ) -> list[float]: ...


class LLMExpert(Expert):
    """A calibration-scored AI panel member."""

    def __init__(
        self,
        name: str,
        base_model: str,
        client: CompletionClient,
        role: str | None = None,
        config: dict | None = None,
    ) -> None:
        super().__init__(name, kind="ai", base_model=base_model, role=role, config=config)
        self._client = client

    def answer_seed(self, question: SeedQuestion, quantile_levels: Sequence[float]) -> QuantileAnswer:
        q = self._client.seed_quantiles(question.text, quantile_levels, self.role)
        return QuantileAnswer(seed_id=question.id, quantiles=tuple(float(x) for x in q))

    def answer_target(self, target: ElicitationTarget) -> DistributionAnswer:
        states = tuple(getattr(target, "states", ()))
        probs = self._client.target_distribution(target.node, target.describe(), states, self.role)
        return DistributionAnswer(
            node=target.node,
            parent_config=tuple(getattr(target, "parent_config", ())),
            probabilities=normalize(probs),
        )


class ScriptedCompletionClient:
    """A deterministic fake for tests. Canned answers keyed by seed text / node."""

    def __init__(
        self,
        seed_answers: dict[str, Sequence[float]],
        target_answers: dict[str, Sequence[float]],
    ) -> None:
        self._seed = {k: list(v) for k, v in seed_answers.items()}
        self._target = {k: list(v) for k, v in target_answers.items()}

    def seed_quantiles(self, question_text, quantile_levels, role) -> list[float]:
        return list(self._seed[question_text])

    def target_distribution(self, node, prompt, states, role) -> list[float]:
        return list(self._target[node])


class OpenAICompletionClient:
    """A thin OpenAI adapter (requires a key; not exercised by unit tests).

    The prompt construction and JSON parsing below *are* unit-tested; only the
    network call is left for integration use.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._api_key = api_key
        self.model = model

    @staticmethod
    def build_distribution_prompt(prompt: str, states: Sequence[str], role: str | None) -> str:
        persona = f"Adopt the role of a {role}. " if role else ""
        return (
            f"{persona}Estimate {prompt}. Respond with ONLY a JSON array of "
            f"{len(states)} probabilities (summing to 1) for states "
            f"{list(states)}."
        )

    @staticmethod
    def parse_float_array(content: str) -> list[float]:
        """Extract the first JSON array of numbers from a model response."""
        match = re.search(r"\[[^\]]*\]", content, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON array found in response: {content!r}")
        values = json.loads(match.group(0))
        return [float(x) for x in values]

    def _complete(self, prompt: str) -> str:  # pragma: no cover - network call
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""

    def seed_quantiles(self, question_text, quantile_levels, role):  # pragma: no cover
        prompt = (
            f"{f'Adopt the role of a {role}. ' if role else ''}For the quantity: "
            f"{question_text}. Give the {list(quantile_levels)} quantiles as a JSON array."
        )
        return self.parse_float_array(self._complete(prompt))

    def target_distribution(self, node, prompt, states, role):  # pragma: no cover
        return self.parse_float_array(self._complete(self.build_distribution_prompt(prompt, states, role)))
