"""News-headline → BN-node-state translator (OpenAI-backed).

This is the *translation layer* in the two-layer architecture: it reads a
free-text headline and returns the node/state assignments it implies,
which are then fed into the Bayesian network as observed evidence. In
production this layer could be any NLP/LLM pipeline; here it is a single
OpenAI chat call with JSON-schema structured output.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .network import SCENARIO_NARRATIVES, STATES

DEFAULT_MODEL = "gpt-4o-mini"


class TranslatorError(RuntimeError):
    """Raised for any translator-side failure the UI should surface."""


@dataclass
class TranslatorAssignment:
    """A single node/state assignment proposed by the translator."""

    node: str
    state: str
    reason: str


@dataclass
class TranslatorResult:
    """Parsed, validated output of a translator call."""

    headline: str
    assignments: List[TranslatorAssignment]
    rationale: str
    model: str

    def as_evidence_dict(self) -> Dict[str, str]:
        """Flatten assignments to a {node: state} dict (later wins on conflict)."""
        out: Dict[str, str] = {}
        for a in self.assignments:
            out[a.node] = a.state
        return out


# ---------------------------------------------------------------------------
# Schema and prompt construction
# ---------------------------------------------------------------------------


def _node_state_enum_schema() -> Dict:
    """JSON schema describing the allowed node set (state validated post-hoc).

    OpenAI's `strict` structured-output mode does not support per-item
    conditional enums (state enum keyed on node value), so we constrain
    `node` to a fixed enum and validate `state` against STATES[node] in
    Python after the call returns.
    """
    node_names = list(STATES.keys())
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["assignments", "overall_rationale"],
        "properties": {
            "assignments": {
                "type": "array",
                "description": (
                    "Node/state pairs this headline constrains. Include "
                    "ONLY nodes the headline speaks to directly or by "
                    "strong implication; omit everything else."
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["node", "state", "reason"],
                    "properties": {
                        "node": {"type": "string", "enum": node_names},
                        "state": {"type": "string"},
                        "reason": {
                            "type": "string",
                            "description": "One short sentence explaining this assignment.",
                        },
                    },
                },
            },
            "overall_rationale": {
                "type": "string",
                "description": "One or two sentences summarising the translation as a whole.",
            },
        },
    }


def _system_prompt() -> str:
    """Build the system prompt describing the BN schema to the model."""
    lines = [
        "You are the translation layer between geopolitical news and a "
        "Bayesian network that tracks three Strait-of-Hormuz scenarios:",
        "",
    ]
    for scenario, narrative in SCENARIO_NARRATIVES.items():
        lines.append(f"  - {scenario}: {narrative}")
    lines += [
        "",
        "The network has the following nodes and allowed states. You MUST "
        "use these exact node names and choose a state from the listed "
        "options for that node.",
        "",
    ]
    for node, states in STATES.items():
        if node == "Scenario":
            continue  # Scenario is the terminal node, never set as evidence.
        lines.append(f"  - {node}: {states}")
    lines += [
        "",
        "Given one news headline, output a JSON object with:",
        "  - assignments: list of {node, state, reason}. Include only nodes "
        "the headline directly speaks to or strongly implies. Typical "
        "headlines map to 1-3 assignments. Do NOT invent assignments.",
        "  - overall_rationale: one or two sentences summarising your read.",
        "",
        "Do not set the 'Scenario' node; it is the terminal node to be "
        "inferred, not observed. Prefer the most specific state that is "
        "clearly supported; if a headline is ambiguous on a node, omit it.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """True if the translator can be called (API key present)."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def translate_headline(
    headline: str,
    *,
    model: str = DEFAULT_MODEL,
    client=None,
) -> TranslatorResult:
    """Translate a free-text headline into BN node assignments.

    Raises `TranslatorError` if the API key is missing, the call fails,
    or the response contains invalid node/state pairs.
    """
    headline = headline.strip()
    if not headline:
        raise TranslatorError("Headline is empty.")
    if not is_available() and client is None:
        raise TranslatorError(
            "OPENAI_API_KEY is not set. Use the manual node picker, or "
            "export the key and retry."
        )

    if client is None:
        from openai import OpenAI

        client = OpenAI()

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": f"Headline: {headline}"},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "bn_translation",
                    "strict": True,
                    "schema": _node_state_enum_schema(),
                },
            },
        )
    except Exception as exc:  # network, auth, rate limit, etc.
        raise TranslatorError(f"OpenAI call failed: {exc}") from exc

    try:
        payload = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, IndexError, AttributeError) as exc:
        raise TranslatorError(f"Malformed translator response: {exc}") from exc

    assignments: List[TranslatorAssignment] = []
    for item in payload.get("assignments", []):
        node = item.get("node")
        state = item.get("state")
        reason = item.get("reason", "")
        if node not in STATES:
            raise TranslatorError(f"Translator returned unknown node: {node!r}")
        if node == "Scenario":
            # Defensive: the prompt forbids this, but drop it if it leaks.
            continue
        if state not in STATES[node]:
            raise TranslatorError(
                f"Translator returned invalid state {state!r} for node "
                f"{node!r}; valid: {STATES[node]}"
            )
        assignments.append(TranslatorAssignment(node=node, state=state, reason=reason))

    return TranslatorResult(
        headline=headline,
        assignments=assignments,
        rationale=payload.get("overall_rationale", ""),
        model=model,
    )


__all__ = [
    "DEFAULT_MODEL",
    "TranslatorAssignment",
    "TranslatorError",
    "TranslatorResult",
    "is_available",
    "translate_headline",
]
