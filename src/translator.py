"""News-headline → BN-node-state translator with pluggable providers.

Provider preference order:
  1. ``claude-code`` — via ``claude-agent-sdk``, reusing the Claude Code
     CLI's logged-in session (no API key needed, billed against the
     Max subscription).
  2. ``openai``      — via the ``openai`` SDK and ``OPENAI_API_KEY``.

``translate_headline`` picks the first available provider unless the
caller passes ``provider=...`` explicitly. If neither is available the
UI degrades to a manual node/state picker.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from .network import SCENARIO_NARRATIVES, STATES

Provider = Literal["claude-code", "openai"]

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
CLAUDE_DEFAULT_MODEL = "claude-haiku-4-5"


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
    provider: Provider

    def as_evidence_dict(self) -> Dict[str, str]:
        """Flatten assignments to a {node: state} dict (later wins on conflict)."""
        out: Dict[str, str] = {}
        for a in self.assignments:
            out[a.node] = a.state
        return out


# ---------------------------------------------------------------------------
# Schema and prompt construction (shared across providers)
# ---------------------------------------------------------------------------


def _node_state_enum_schema() -> Dict:
    """JSON schema for the translator output.

    OpenAI's `strict` mode does not support per-item conditional enums
    (state enum keyed on node value), so we constrain `node` to a fixed
    enum and validate `state` against STATES[node] in Python.
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
                "description": "One or two sentences summarising the translation.",
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
            continue
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
        "Output ONLY the JSON object, with no prose before or after it.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Provider availability
# ---------------------------------------------------------------------------


def _claude_code_available() -> bool:
    """True if the Claude Code CLI is installed and the SDK is importable."""
    if shutil.which("claude") is None:
        return False
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError:
        return False
    return True


def _openai_available() -> bool:
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True


def available_providers() -> List[Provider]:
    """Providers the current environment can actually call, in preference order."""
    providers: List[Provider] = []
    if _claude_code_available():
        providers.append("claude-code")
    if _openai_available():
        providers.append("openai")
    return providers


def is_available() -> bool:
    """True if at least one provider can be called."""
    return bool(available_providers())


# ---------------------------------------------------------------------------
# Shared response validation
# ---------------------------------------------------------------------------


def _validate_payload(payload: Dict) -> tuple[List[TranslatorAssignment], str]:
    assignments: List[TranslatorAssignment] = []
    for item in payload.get("assignments", []):
        node = item.get("node")
        state = item.get("state")
        reason = item.get("reason", "")
        if node not in STATES:
            raise TranslatorError(f"Translator returned unknown node: {node!r}")
        if node == "Scenario":
            continue  # drop any Scenario leak
        if state not in STATES[node]:
            raise TranslatorError(
                f"Translator returned invalid state {state!r} for node "
                f"{node!r}; valid: {STATES[node]}"
            )
        assignments.append(TranslatorAssignment(node=node, state=state, reason=reason))
    return assignments, payload.get("overall_rationale", "")


def _extract_json_block(text: str) -> Dict:
    """Best-effort JSON extraction from a model's text response."""
    text = text.strip()
    # Fast path: the whole response is JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: grab the outermost JSON object via a greedy brace match.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise TranslatorError("No JSON object found in translator response.")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise TranslatorError(f"Malformed translator JSON: {exc}") from exc


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------


def _translate_openai(
    headline: str, *, model: str = OPENAI_DEFAULT_MODEL, client=None
) -> TranslatorResult:
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
    except Exception as exc:
        raise TranslatorError(f"OpenAI call failed: {exc}") from exc

    try:
        payload = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, IndexError, AttributeError) as exc:
        raise TranslatorError(f"Malformed OpenAI response: {exc}") from exc

    assignments, rationale = _validate_payload(payload)
    return TranslatorResult(
        headline=headline,
        assignments=assignments,
        rationale=rationale,
        model=model,
        provider="openai",
    )


# ---------------------------------------------------------------------------
# Claude Code backend (via claude-agent-sdk)
# ---------------------------------------------------------------------------


async def _claude_code_collect(prompt: str, *, model: str) -> str:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        TextBlock,
        query,
    )

    options = ClaudeAgentOptions(
        system_prompt=_system_prompt(),
        model=model,
        allowed_tools=[],              # pure LLM call, no tools
        max_turns=1,
        permission_mode="bypassPermissions",
    )
    chunks: List[str] = []
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
    return "".join(chunks).strip()


def _translate_claude_code(
    headline: str, *, model: str = CLAUDE_DEFAULT_MODEL
) -> TranslatorResult:
    prompt = (
        f"Headline: {headline}\n\n"
        "Respond with the JSON object described in the system prompt, and nothing else."
    )
    try:
        text = asyncio.run(_claude_code_collect(prompt, model=model))
    except Exception as exc:
        raise TranslatorError(f"Claude Code call failed: {exc}") from exc

    payload = _extract_json_block(text)
    assignments, rationale = _validate_payload(payload)
    return TranslatorResult(
        headline=headline,
        assignments=assignments,
        rationale=rationale,
        model=model,
        provider="claude-code",
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def translate_headline(
    headline: str,
    *,
    provider: Optional[Provider] = None,
    client=None,
) -> TranslatorResult:
    """Translate a headline via the preferred (or requested) provider.

    Passing ``client=<openai-like>`` forces the OpenAI path and is used
    exclusively by the test suite; production callers should leave both
    ``provider`` and ``client`` unset and let the dispatcher pick.
    """
    headline = headline.strip()
    if not headline:
        raise TranslatorError("Headline is empty.")

    if client is not None:
        return _translate_openai(headline, client=client)

    chosen: Optional[Provider]
    if provider is not None:
        chosen = provider
    else:
        providers = available_providers()
        chosen = providers[0] if providers else None

    if chosen == "claude-code":
        return _translate_claude_code(headline)
    if chosen == "openai":
        return _translate_openai(headline)
    raise TranslatorError(
        "No translator provider is available. Install Claude Code and sign in, "
        "or export OPENAI_API_KEY."
    )


__all__ = [
    "CLAUDE_DEFAULT_MODEL",
    "OPENAI_DEFAULT_MODEL",
    "Provider",
    "TranslatorAssignment",
    "TranslatorError",
    "TranslatorResult",
    "available_providers",
    "is_available",
    "translate_headline",
]
