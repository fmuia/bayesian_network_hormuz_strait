"""Elicitation clients — the LLM boundary for whole-network elicitation.

A client answers two things, each in ONE call (so a whole-network run costs
~one call per node per agent, not per column):

* ``seed_quantiles`` — quantiles for the calibration seeds, plus a rationale.
* ``node_cpt`` — a distribution for every parent configuration of a node, plus a
  rationale.

The real :class:`ClaudeClient` reuses the same ``claude-agent-sdk`` path the
translator uses (the logged-in Claude Code session — no API key, billed to the
Max subscription). :class:`OpenAIClient` is the optional second model (needs
``OPENAI_API_KEY``) used for multi-model decorrelation. :class:`ScriptedClient`
is the deterministic fake the tests use. Only the network calls are left
untested (``# pragma: no cover``).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
from typing import Callable, Protocol, Sequence

import numpy as np

from ..protocols.base import SeedQuestion

CLAUDE_DEFAULT_MODEL = "claude-sonnet-4-5"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"

# A parent configuration is a tuple of parent state names ( () for a root ).
Config = tuple[str, ...]


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object in response: {text[:200]!r}")
        return json.loads(match.group(0))


def _normalize(values: Sequence[float], n: int) -> list[float]:
    arr = np.clip(np.asarray(values, dtype=float), 0.0, None)
    if arr.size != n or arr.sum() <= 0:
        return [1.0 / n] * n
    return list(arr / arr.sum())


class ElicitationClient(Protocol):
    name: str

    def seed_quantiles(
        self, seeds: Sequence[SeedQuestion], quantile_levels: Sequence[float], role: str | None
    ) -> tuple[dict[str, list[float]], dict[str, str], str]: ...  # (quantiles, basis, rationale)

    def node_cpt(
        self,
        node: str,
        states: Sequence[str],
        parents: Sequence[str],
        parent_configs: Sequence[Config],
        role: str | None,
    ) -> tuple[dict[Config, list[float]], str]: ...

    def propose_roles(
        self, node: str, states: Sequence[str], parents: Sequence[str], n: int
    ) -> list[str]: ...


# --------------------------------------------------------------------------- #
# Prompts (shared by the real clients)
# --------------------------------------------------------------------------- #


def _persona(role: str | None) -> str:
    """The role framing. Agents are told they are consulted *independently* (no
    cross-agent anchoring → counters sycophancy/false consensus, methodology §8.5).
    Skeptic and base-rate roles get an extra directive so the role does real work."""
    if not role:
        return "You are a calibrated domain expert, consulted independently. "
    base = f"You are acting as a {role} on a panel whose members are consulted independently. "
    low = role.lower()
    if any(k in low for k in ("skeptic", "red", "adversar", "devil", "contrarian")):
        base += "Actively challenge optimistic or consensus assumptions and argue for wider uncertainty. "
    elif any(k in low for k in ("base rate", "base-rate", "outside", "forecaster", "statistician", "actuar")):
        base += "Anchor firmly on reference-class base rates before any case-specific adjustment. "
    return base


def _reasoning_directive(effort: str) -> str:
    """How much to deliberate before committing — and *how* to reason. Bakes in the
    debiasing protocol (methodology §8): outside view first (counters anchoring /
    base-rate neglect) and consider-the-opposite (counters one-sided overconfidence).
    Bounds reasoning time and forces a conclusion within budget."""
    depth = {
        "brief": "In a sentence or two,",
        "balanced": "Concisely,",
        "thorough": "Carefully but without rambling,",
    }.get(effort, "Concisely,")
    return (
        f"{depth} reason from the outside view first (reference-class base rates), then adjust for the "
        f"specifics. Before committing, note one reason your estimate could be too high and one too low. "
        f"Always finish within this budget and output the final JSON — do not deliberate indefinitely."
    )


# Appended when an agent runs past its soft time budget: ask it to conclude now.
URGENT_SUFFIX = (
    "NOTE: you are over your time budget. Stop reasoning immediately and output "
    "your single best-estimate JSON right now."
)


def build_urgent_prompt(prompt: str, partial_reasoning: str) -> str:
    """The 'conclude now' re-ask, carrying the agent's reasoning-so-far as context
    so the first attempt's work is reused rather than wasted."""
    context = (
        f"\n\nYour reasoning so far (build on this, do not restart):\n{partial_reasoning.strip()}\n"
        if partial_reasoning.strip()
        else ""
    )
    return f"{prompt}{context}\n\n{URGENT_SUFFIX}"


def build_seed_prompt(
    seeds: Sequence[SeedQuestion], levels: Sequence[float], role: str | None, reasoning: str = "balanced"
) -> str:
    items = "\n".join(f'  - id "{s.id}": {s.text}' for s in seeds)
    inner_pct = round((max(levels) - min(levels)) * 100)
    return (
        f"{_persona(role)}{_reasoning_directive(reasoning)}\n"
        f"These are calibration questions. For each, give your {list(levels)} quantiles for the unknown "
        f"quantity — a range, not a point answer.\n"
        f"Calibrate honestly: your outer [{min(levels)}, {max(levels)}] quantiles should be wide enough "
        f"that the true value falls inside them about {inner_pct}% of the time across many such questions. "
        f"Most experts and models make these ranges TOO NARROW — when unsure, widen rather than guess. "
        f"Estimate from first principles; do NOT answer from a specific figure you happen to recall.\n"
        f'For each answer also report "basis": "recall" if your number comes from a specific figure or '
        f'source you remember, otherwise "estimate". Be honest — recalled answers are DISCARDED, not '
        f"penalised, because they test memory rather than calibration.\n"
        f"{items}\n\n"
        'Respond ONLY with JSON: {"answers":[{"id":"...","quantiles":[..],"basis":"estimate|recall"}],'
        '"rationale":"one or two sentences"}.'
    )


def build_node_prompt(
    node: str,
    states: Sequence[str],
    parents: Sequence[str],
    parent_configs: Sequence[Config],
    role: str | None,
    reasoning: str = "balanced",
) -> str:
    cfgs = "\n".join(f"  - {list(c)}" for c in parent_configs) if parents else "  - [] (root node)"
    return (
        # TODO(pack-separation): "Strait-of-Hormuz crisis dynamics" is scenario-specific
        # prompt text — read the domain from the active pack's TranslatorProfile.domain.
        f"{_persona(role)}{_reasoning_directive(reasoning)}\n"
        f"Elicit the conditional probability table for the node below in a Bayesian network modelling "
        f"Strait-of-Hormuz crisis dynamics.\n"
        f"Node: {node}\nStates (in order): {list(states)}\nParents: {list(parents)}\n\n"
        f"Method — work through this, then output JSON:\n"
        f"1. Base rate: the marginal distribution over the states ignoring the parents (reference class).\n"
        f"2. Parent effects: for each parent, which states it pushes up or down.\n"
        f"3. Coherence: as the parents move toward more escalatory/extreme values, the corresponding "
        f"extreme states must NOT become less likely — keep the table monotone and internally consistent.\n"
        f"4. Reserve mass for surprise: avoid 0 or 1 unless logically necessary; leave some probability on "
        f"off-nominal outcomes (counter overconfidence).\n"
        f"5. Reason from causal structure, not from any specific remembered event.\n\n"
        f"For EACH parent configuration below, give a probability distribution over the states (in the "
        f"listed order, summing to 1):\n{cfgs}\n\n"
        'Respond ONLY with JSON: {"columns":[{"config":[..],"probs":[..]}],'
        '"rationale":"one sentence on the main driver"}.'
    )


def _parse_seed_response(
    payload: dict, seeds: Sequence[SeedQuestion]
) -> tuple[dict[str, list[float]], dict[str, str], str]:
    """Returns (quantiles_by_seed, basis_by_seed, rationale). ``basis`` is the
    agent's self-report 'estimate' | 'recall' per seed; a missing/unknown value
    defaults to 'estimate' (benefit of the doubt — never over-discards)."""
    answers: dict[str, list[float]] = {}
    basis: dict[str, str] = {}
    for item in payload.get("answers", []):
        sid = item.get("id")
        q = item.get("quantiles")
        if sid is not None and q:
            answers[str(sid)] = [float(x) for x in q]
            basis[str(sid)] = "recall" if str(item.get("basis", "")).strip().lower() == "recall" else "estimate"
    return answers, basis, str(payload.get("rationale", ""))


def _parse_node_response(
    payload: dict, states: Sequence[str], parent_configs: Sequence[Config]
) -> tuple[dict[Config, list[float]], str]:
    by_config = {tuple(item.get("config", [])): item.get("probs", []) for item in payload.get("columns", [])}
    n = len(states)
    columns: dict[Config, list[float]] = {}
    for cfg in parent_configs:
        columns[cfg] = _normalize(by_config.get(cfg, []), n)
    return columns, str(payload.get("rationale", ""))


def build_roles_prompt(node: str, states: Sequence[str], parents: Sequence[str], n: int) -> str:
    outside = " and at least one base-rate / outside-view forecaster" if n >= 3 else ""
    return (
        # TODO(pack-separation): "Strait-of-Hormuz crisis" is scenario-specific prompt
        # text — read the domain from the active pack's TranslatorProfile.domain.
        f"Recruit an expert panel to elicit the conditional probability table for the node '{node}' "
        f"(states {list(states)}, parents {list(parents)}) in a Strait-of-Hormuz crisis Bayesian "
        f"network. Propose {n} DISTINCT, node-appropriate expert roles bringing different disciplinary "
        f"lenses (e.g. naval analyst, maritime-insurance underwriter, energy economist). Include at "
        f"least one red-team skeptic{outside}. Output role TITLES ONLY — no descriptions, ≤6 words each.\n"
        'Respond ONLY with JSON: {"roles":["...", "..."]}.'
    )


def _parse_roles(payload: dict, n: int) -> list[str]:
    roles = [str(r) for r in payload.get("roles", [])][:n]
    while len(roles) < n:
        roles.append(f"domain expert {len(roles) + 1}")
    return roles


# --------------------------------------------------------------------------- #
# Real clients
# --------------------------------------------------------------------------- #


class ClaudeClient:
    """Claude via the logged-in Claude Code session (no API key)."""

    provider = "claude-code"

    def __init__(
        self,
        model: str = CLAUDE_DEFAULT_MODEL,
        time_limit: float = 45.0,
        hard_timeout: float = 180.0,
        reasoning: str = "balanced",
    ) -> None:
        self.model = model
        self.name = model
        self.time_limit = time_limit
        self.hard_timeout = hard_timeout
        self.reasoning = reasoning

    @staticmethod
    def available() -> bool:
        if shutil.which("claude") is None:
            return False
        return importlib.util.find_spec("claude_agent_sdk") is not None

    async def _stream(self, prompt: str, answer: list[str], reasoning: list[str]) -> None:  # pragma: no cover - network
        """Stream output, appending answer text and thinking into the passed
        buffers as they arrive — so a soft-timeout cancel still leaves the partial
        work available for the urgent re-ask."""
        from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, ThinkingBlock, query

        options = ClaudeAgentOptions(
            system_prompt="Output only the requested JSON object, no prose.",
            model=self.model,
            allowed_tools=[],
            max_turns=1,
            permission_mode="bypassPermissions",
        )
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        answer.append(block.text)
                    elif isinstance(block, ThinkingBlock):
                        reasoning.append(block.thinking)

    def _complete(self, prompt: str) -> str:  # pragma: no cover - network
        import asyncio

        answer: list[str] = []
        reasoning: list[str] = []
        try:
            asyncio.run(asyncio.wait_for(self._stream(prompt, answer, reasoning), timeout=self.time_limit))
            return "".join(answer).strip()
        except asyncio.TimeoutError:
            # Over the soft budget: re-ask to conclude now, reusing the work so far.
            partial = ("".join(reasoning) + "\n" + "".join(answer)).strip()
            urgent = build_urgent_prompt(prompt, partial)
            answer2: list[str] = []
            asyncio.run(asyncio.wait_for(self._stream(urgent, answer2, []), timeout=self.hard_timeout))
            return "".join(answer2).strip()

    def seed_quantiles(self, seeds, levels, role):  # pragma: no cover - network
        payload = _extract_json(self._complete(build_seed_prompt(seeds, levels, role, self.reasoning)))
        return _parse_seed_response(payload, seeds)

    def node_cpt(self, node, states, parents, parent_configs, role):  # pragma: no cover - network
        prompt = build_node_prompt(node, states, parents, parent_configs, role, self.reasoning)
        return _parse_node_response(_extract_json(self._complete(prompt)), states, parent_configs)

    def propose_roles(self, node, states, parents, n):  # pragma: no cover - network
        return _parse_roles(_extract_json(self._complete(build_roles_prompt(node, states, parents, n))), n)


class OpenAIClient:
    """OpenAI via ``OPENAI_API_KEY`` — the optional second base model."""

    provider = "openai"

    def __init__(
        self,
        model: str = OPENAI_DEFAULT_MODEL,
        time_limit: float = 45.0,
        hard_timeout: float = 180.0,
        reasoning: str = "balanced",
    ) -> None:
        self.model = model
        self.name = model
        self.time_limit = time_limit
        self.hard_timeout = hard_timeout
        self.reasoning = reasoning

    @staticmethod
    def available() -> bool:
        if not os.environ.get("OPENAI_API_KEY"):
            return False
        return importlib.util.find_spec("openai") is not None

    def _call(self, prompt: str, timeout: float) -> str:  # pragma: no cover - network
        from openai import OpenAI

        client = OpenAI(timeout=timeout)
        resp = client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Output only the requested JSON object, no prose."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""

    def _complete(self, prompt: str) -> str:  # pragma: no cover - network
        from openai import APITimeoutError

        try:
            return self._call(prompt, self.time_limit)
        except APITimeoutError:
            # Over the soft budget: re-ask to conclude now (non-streamed: no partial).
            return self._call(build_urgent_prompt(prompt, ""), self.hard_timeout)

    def seed_quantiles(self, seeds, levels, role):  # pragma: no cover - network
        prompt = build_seed_prompt(seeds, levels, role, self.reasoning)
        return _parse_seed_response(_extract_json(self._complete(prompt)), seeds)

    def node_cpt(self, node, states, parents, parent_configs, role):  # pragma: no cover - network
        prompt = build_node_prompt(node, states, parents, parent_configs, role, self.reasoning)
        return _parse_node_response(_extract_json(self._complete(prompt)), states, parent_configs)

    def propose_roles(self, node, states, parents, n):  # pragma: no cover - network
        return _parse_roles(_extract_json(self._complete(build_roles_prompt(node, states, parents, n))), n)


# --------------------------------------------------------------------------- #
# Deterministic fake (tests / offline)
# --------------------------------------------------------------------------- #


class ScriptedClient:
    """Deterministic client. ``seed_answers`` maps seed id -> quantiles;
    ``node_fn(node, config, states)`` returns an (unnormalised) vector."""

    provider = "scripted"

    def __init__(
        self,
        name: str,
        seed_answers: dict[str, Sequence[float]],
        node_fn: Callable[[str, Config, Sequence[str]], Sequence[float]],
        rationale: str = "Illustrative reasoning for this node.",
        recall_ids: set[str] | None = None,
    ) -> None:
        self.name = name
        self._seed = {k: list(v) for k, v in seed_answers.items()}
        self._node_fn = node_fn
        self._rationale = rationale
        self._recall_ids = set(recall_ids or ())  # seeds this fake "admits" it recalled

    def seed_quantiles(self, seeds, levels, role):
        answers = {s.id: list(self._seed[s.id]) for s in seeds}
        basis = {s.id: ("recall" if s.id in self._recall_ids else "estimate") for s in seeds}
        return answers, basis, self._rationale

    def node_cpt(self, node, states, parents, parent_configs, role):
        columns = {cfg: _normalize(self._node_fn(node, cfg, states), len(states)) for cfg in parent_configs}
        return columns, f"{self._rationale} (role={role})"

    def propose_roles(self, node, states, parents, n):
        base = [f"{node} analyst", "maritime-risk underwriter", "energy economist", "red-team skeptic"]
        return [base[i % len(base)] for i in range(n)]


def available_providers() -> list[str]:
    """Which real elicitation providers can be called right now."""
    out: list[str] = []
    if ClaudeClient.available():
        out.append("claude-code")
    if OpenAIClient.available():
        out.append("openai")
    return out


def make_client(
    provider: str,
    model: str | None = None,
    time_limit: float = 45.0,
    hard_timeout: float = 180.0,
    reasoning: str = "balanced",
):
    """Construct a real client for a provider."""
    if provider == "claude-code":
        return ClaudeClient(model or CLAUDE_DEFAULT_MODEL, time_limit, hard_timeout, reasoning)
    if provider == "openai":
        return OpenAIClient(model or OPENAI_DEFAULT_MODEL, time_limit, hard_timeout, reasoning)
    raise ValueError(f"unknown provider {provider!r}")


__all__ = [
    "ElicitationClient",
    "ClaudeClient",
    "OpenAIClient",
    "ScriptedClient",
    "available_providers",
    "make_client",
    "build_seed_prompt",
    "build_node_prompt",
    "CLAUDE_DEFAULT_MODEL",
    "OPENAI_DEFAULT_MODEL",
]
