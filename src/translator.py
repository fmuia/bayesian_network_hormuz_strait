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
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

from .network import SCENARIO_NARRATIVES, STATES

Provider = Literal["claude-code", "openai", "fake"]

SourceType = Literal[
    "wire_service", "commercial_press", "state_media",
    "analyst_note", "social_media", "unknown",
]

# Default per-source-type credibility weight w ∈ [0, 1] (Plan 2 §B1), applied as
# a power discount on the likelihood ratios (ε ← ε**w): w=1 full evidence, w=0
# no information. Per-source overrides + edit history come in T10 (B1b).
SOURCE_TYPE_CREDIBILITY: Dict[str, float] = {
    "wire_service": 1.0,
    "commercial_press": 0.8,
    "analyst_note": 0.7,
    "state_media": 0.3,
    "social_media": 0.2,
    "unknown": 0.5,
}

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
CLAUDE_DEFAULT_MODEL = "claude-opus-4-8"

# A1 likelihood-ratio semantics: per-state ε ∈ (0, 1] with the best-supported
# state pinned to 1.0 (NOT a probability distribution; does not sum to 1).
_EPS_FLOOR = 0.01  # default for states the model did not mention ("ruled out")
_EPS_TOL = 1e-6

# Offline dev/test provider: deterministic fixtures, no network. Selected via
# provider="fake", TRANSLATOR_PROVIDER=fake, or the dashboard dev toggle; kept
# OUT of available_providers() so it is never auto-selected over a real backend.
_FAKE_FIXTURES_DIR = Path(
    os.environ.get(
        "TRANSLATOR_FAKE_DIR",
        str(Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "translator"),
    )
)

# Callback type: fn(stage, detail) where stage is a short machine tag and
# detail is a human-readable sentence. Used by the UI to stream progress.
from typing import Callable  # noqa: E402
StepCallback = Callable[[str, str], None]


class TranslatorError(RuntimeError):
    """Raised for any translator-side failure the UI should surface."""


@dataclass
class TranslatorAssignment:
    """A single node/state assignment proposed by the translator.

    ``state_probs`` holds the per-state **likelihood ratios** ε ∈ (0, 1] (A1
    semantics): how plausible the article is given each state, scaled so the
    best-supported state = 1.0. It is NOT a probability distribution and does
    not sum to 1. (Field name retained pending the A2 schema cleanup.)
    """

    node: str
    state: str
    reason: str
    state_probs: Dict[str, float]


@dataclass
class TranslatorResult:
    """Parsed, validated output of a translator call."""

    headline: str
    assignments: List[TranslatorAssignment]
    rationale: str
    model: str
    provider: Provider
    raw_response: str = ""  # verbatim model output (for debug / audit)
    semantics_version: str = "likelihood-ratio"  # A1; pre-A1 records: "pre-A1-posterior"

    def as_evidence_dict(self) -> Dict[str, str]:
        """Flatten assignments to a {node: state} dict (later wins on conflict)."""
        out: Dict[str, str] = {}
        for a in self.assignments:
            out[a.node] = a.state
        return out


@dataclass
class Article:
    """An article (or bare headline) to translate (Plan 2 B1a).

    Tolerates missing ``body``/``url``/``published_at`` — a bare headline is the
    degenerate case (``translate_headline`` builds one). The prompt is told which
    fields are present so it can weight the body over the headline when they
    disagree, and downgrade confidence when working from a headline alone.
    """

    headline: str
    lede: str = ""
    body: str = ""
    source: str = ""
    source_type: SourceType = "unknown"
    url: str = ""
    published_at: str = ""
    language: str = "en"


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
                    "required": ["node", "state", "reason", "state_probs"],
                    "properties": {
                        "node": {"type": "string", "enum": node_names},
                        "state": {"type": "string"},
                        "reason": {
                            "type": "string",
                            "description": "One short sentence explaining this assignment.",
                        },
                        "state_probs": {
                            "type": "array",
                            "description": (
                                "Relative likelihood of the article for each allowed "
                                "state (how plausible the article is if that state were "
                                "the true one), scaled so the single best-supported state "
                                "= 1.0 and the rest are fractions in (0, 1]. NOT a "
                                "probability distribution; does NOT sum to 1."
                            ),
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["state", "value"],
                                "properties": {
                                    "state": {"type": "string"},
                                    "value": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                        "description": (
                                            "Relative likelihood ε for this state in (0, 1]; "
                                            "the best-supported state in the array = 1.0."
                                        ),
                                    },
                                },
                            },
                            "minItems": 2,
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


def _states_hash() -> str:
    """Short stable hash of the node taxonomy (``STATES``).

    Embedded in the prompt and recorded so drift between ``network.py`` and a
    frozen/externalised prompt is detectable (used by D1/T08 prompt versioning).
    """
    canonical = json.dumps(STATES, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


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
        f"(Node-taxonomy snapshot: schema hash {_states_hash()}. Use ONLY the node "
        "names listed above; any other name is rejected.)",
        "",
        "You are given an article (headline, and optionally lede, body, and source "
        "metadata). Weight the body over the headline when they disagree, and keep "
        "confidence moderate when only a headline is available. Output a JSON object with:",
        "  - assignments: list of {node, state, reason, state_probs}. Include only nodes "
        "the headline directly speaks to or strongly implies. Typical "
        "headlines map to 1-3 assignments. Do NOT invent assignments.",
        "    For each assignment, state_probs is a list of {state, value} objects giving a",
        "    RELATIVE LIKELIHOOD for every",
        "    allowed state of the node: how plausible this article is if that state were",
        "    the true one. Scale them so the single best-supported state = 1.0 and the",
        "    others are fractions in (0, 1]. These are NOT probabilities and must NOT sum",
        "    to 1. Include ALL allowed states; for a state the article essentially rules",
        "    out, use a small floor like 0.01 (never 0). E.g. a 3-state node: 1.0, 0.3, 0.05.",
        "  - overall_rationale: one or two sentences summarising your read.",
        "",
        "Do not set the 'Scenario' node; it is the terminal node to be "
        "inferred, not observed. Prefer the most specific state that is "
        "clearly supported; if a headline is ambiguous on a node, omit it.",
        "Output ONLY the JSON object, with no prose before or after it.",
    ]
    return "\n".join(lines)


def _article_user_content(article: Article) -> str:
    """Render the user-message content for an article (B1a)."""
    parts = [f"Headline: {article.headline}"]
    if article.lede:
        parts.append(f"Lede: {article.lede}")
    if article.body:
        parts.append(f"Body:\n{article.body}")
    meta = []
    if article.source:
        meta.append(f"source={article.source}")
    if article.source_type and article.source_type != "unknown":
        meta.append(f"source_type={article.source_type}")
    if article.published_at:
        meta.append(f"published_at={article.published_at}")
    if meta:
        parts.append("Metadata: " + ", ".join(meta))
    if not article.body:
        parts.append(
            "(No article body provided — work from the headline alone and keep "
            "confidence moderate.)"
        )
    return "\n\n".join(parts)


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
        probs_raw = item.get("state_probs", [])
        if node not in STATES:
            raise TranslatorError(f"Translator returned unknown node: {node!r}")
        if node == "Scenario":
            continue  # drop any Scenario leak
        if state not in STATES[node]:
            raise TranslatorError(
                f"Translator returned invalid state {state!r} for node "
                f"{node!r}; valid: {STATES[node]}"
            )
        probs: Dict[str, float]
        if probs_raw:
            # A2: a single canonical shape — an array of {state, value} objects.
            # The permissive dict / list-of-{prob} / JSON-string branches are
            # gone (C6); nothing is silently coerced (C7).
            if not isinstance(probs_raw, list):
                raise TranslatorError(
                    f"state_probs for node {node!r} must be an array of "
                    f"{{state, value}} objects, got {type(probs_raw).__name__}."
                )
            probs = {}
            for p in probs_raw:
                if not isinstance(p, dict) or "state" not in p or "value" not in p:
                    raise TranslatorError(
                        f"Malformed state_probs item for node {node!r}: {p!r}; "
                        f"expected {{state, value}}."
                    )
                s = p["state"]
                if s not in STATES[node]:
                    raise TranslatorError(
                        f"Invalid state_probs state {s!r} for node {node!r}; "
                        f"valid: {STATES[node]}"
                    )
                try:
                    probs[s] = float(p["value"])
                except (TypeError, ValueError) as exc:
                    raise TranslatorError(
                        f"Non-numeric value for {node}.{s}: {p.get('value')!r}"
                    ) from exc
            # A1 likelihood-ratio semantics: ε ∈ (0, 1], best state pinned to 1.0.
            # Reject an explicit 0 (a state asserted strictly impossible would
            # zero its posterior irrecoverably); floor only states the model did
            # not mention. Errors carry the offending vector.
            for s, v in probs.items():
                if v <= 0.0:
                    raise TranslatorError(
                        f"Non-positive likelihood {v!r} for {node}.{s} "
                        f"(vector {probs}); use a small floor (e.g. 0.01) for "
                        f"'essentially ruled out', never 0."
                    )
                if v > 1.0 + _EPS_TOL:
                    raise TranslatorError(
                        f"Likelihood {v!r} > 1 for {node}.{s} (vector {probs}); "
                        f"ratios must be in (0, 1] with the best state pinned to 1.0."
                    )
            for s in STATES[node]:
                probs.setdefault(s, _EPS_FLOOR)
            if abs(max(probs.values()) - 1.0) > _EPS_TOL:
                raise TranslatorError(
                    f"Likelihoods for {node} are not max-pinned (no state = 1.0): "
                    f"{probs}. Scale so the best state = 1.0."
                )
        else:
            probs = {s: (1.0 if s == state else _EPS_FLOOR) for s in STATES[node]}
        top_state = max(probs, key=probs.get)
        assignments.append(
            TranslatorAssignment(node=node, state=top_state, reason=reason, state_probs=probs)
        )
    return assignments, payload.get("overall_rationale", "")


def _first_balanced_json_object(text: str) -> Optional[str]:
    """Return the first brace-balanced ``{...}`` substring, or ``None``.

    Tracks string literals and escapes so braces inside JSON strings don't throw
    off the depth count, and stops at the first balanced object so trailing prose
    is ignored. Replaces the old greedy ``\\{.*\\}`` regex (finding C8), which
    over-matched across multiple objects and through string contents.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json_block(text: str) -> Dict:
    """Parse a JSON object from a model's text response (Claude fallback only).

    Used only when the Claude structured-output path returns no
    ``structured_output``; the OpenAI path and the schema-bound Claude path do
    not pass through here.
    """
    text = text.strip()
    try:
        return json.loads(text)  # fast path: the whole response is JSON
    except json.JSONDecodeError:
        pass
    block = _first_balanced_json_object(text)
    if block is None:
        raise TranslatorError("No JSON object found in translator response.")
    try:
        return json.loads(block)
    except json.JSONDecodeError as exc:
        raise TranslatorError(f"Malformed translator JSON: {exc}") from exc


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------


def _translate_openai(
    article: Article,
    *,
    model: str = OPENAI_DEFAULT_MODEL,
    client=None,
    on_step: Optional[StepCallback] = None,
) -> TranslatorResult:
    _emit = on_step or (lambda *_: None)
    if client is None:
        _emit("init", f"Calling OpenAI ({model})…")
        from openai import OpenAI
        client = OpenAI()
    else:
        _emit("init", f"Calling OpenAI ({model})…")
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _article_user_content(article)},
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

    raw = response.choices[0].message.content or ""
    _emit("response", f"Model returned {len(raw)} chars")
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, IndexError, AttributeError) as exc:
        raise TranslatorError(f"Malformed OpenAI response: {exc}") from exc

    assignments, rationale = _validate_payload(payload)
    _emit("validated", f"Validated {len(assignments)} assignment(s)")
    return TranslatorResult(
        headline=article.headline,
        assignments=assignments,
        rationale=rationale,
        model=model,
        provider="openai",
        raw_response=raw,
    )


# ---------------------------------------------------------------------------
# Claude Code backend (via claude-agent-sdk)
# ---------------------------------------------------------------------------


async def _claude_code_collect(
    prompt: str,
    *,
    model: str,
    on_step: Optional[StepCallback] = None,
) -> tuple[str, Optional[Dict]]:
    """Return ``(collected_text, structured_output)``.

    With ``output_format`` set to our JSON schema (A2), the CLI parses and
    validates the model output and hands it back on
    ``ResultMessage.structured_output`` — the schema-bound path that replaces
    regex scraping (C8). ``structured_output`` is ``None`` only if the CLI did
    not produce one, in which case the caller falls back to brace-matching the
    collected text.
    """
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        ThinkingBlock,
        query,
    )

    _emit = on_step or (lambda *_: None)
    opts_kwargs: Dict = dict(
        system_prompt=_system_prompt(),
        model=model,
        allowed_tools=[],              # pure LLM call, no tools
        max_turns=1,
        permission_mode="bypassPermissions",
    )
    if _claude_output_format_enabled():
        # Schema-bound output — only when the CLI is known to support it.
        opts_kwargs["output_format"] = {
            "type": "json_schema", "schema": _node_state_enum_schema(),
        }
    options = ClaudeAgentOptions(**opts_kwargs)
    chunks: List[str] = []
    structured: Optional[Dict] = None
    _emit("init", f"Calling Claude Code ({model})…")
    total_chars = 0
    last_thinking_preview = ""
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
                    total_chars += len(block.text)
                    _emit("response", f"Model returned {total_chars} chars")
                elif isinstance(block, ThinkingBlock):
                    # Surface a short preview of the model's reasoning if
                    # extended thinking is enabled on the CLI side. Guard against
                    # an empty/whitespace thinking block (splitlines() -> []).
                    _t_lines = block.thinking.strip().splitlines()
                    preview = _t_lines[0][:160] if _t_lines else ""
                    if preview and preview != last_thinking_preview:
                        last_thinking_preview = preview
                        _emit("thinking", preview)
        elif isinstance(msg, ResultMessage):
            so = getattr(msg, "structured_output", None)
            if isinstance(so, dict):
                structured = so
    return "".join(chunks).strip(), structured


def _translate_claude_code(
    article: Article,
    *,
    model: str = CLAUDE_DEFAULT_MODEL,
    on_step: Optional[StepCallback] = None,
) -> TranslatorResult:
    _emit = on_step or (lambda *_: None)
    prompt = (
        _article_user_content(article)
        + "\n\nRespond with the JSON object described in the system prompt, "
        "and nothing else."
    )
    try:
        text, structured = asyncio.run(
            _claude_code_collect(prompt, model=model, on_step=on_step)
        )
    except Exception as exc:
        raise TranslatorError(f"Claude Code call failed: {exc}") from exc

    if structured is not None:
        payload = structured  # schema-bound output; no parsing needed
    else:
        _emit("parsing", "Parsing JSON response…")
        try:
            payload = _extract_json_block(text)
        except TranslatorError as exc:
            exc.raw_response = text  # type: ignore[attr-defined]  # surface to UI
            raise

    raw = text or json.dumps(payload)
    try:
        assignments, rationale = _validate_payload(payload)
    except TranslatorError as exc:
        if not getattr(exc, "raw_response", ""):
            exc.raw_response = raw  # type: ignore[attr-defined]
        raise
    _emit("validated", f"Validated {len(assignments)} assignment(s)")
    return TranslatorResult(
        headline=article.headline,
        assignments=assignments,
        rationale=rationale,
        model=model,
        provider="claude-code",
        raw_response=raw,
    )


# ---------------------------------------------------------------------------
# Fake backend (offline dev/test fixtures — no network, deterministic)
# ---------------------------------------------------------------------------


def _load_fake_fixtures() -> List[Dict]:
    """Load every ``*.json`` fixture from the fake-fixtures directory.

    Each fixture is ``{"match": [substr, ...], "kind": "...", "payload": {...}}``.
    A fixture with an empty ``match`` list is the default fallback. A missing
    directory yields an empty list (the fake provider then returns an empty result).
    """
    fixtures: List[Dict] = []
    if not _FAKE_FIXTURES_DIR.is_dir():
        return fixtures
    for path in sorted(_FAKE_FIXTURES_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data["_name"] = path.stem
        fixtures.append(data)
    return fixtures


def _select_fixture(fixtures: List[Dict], headline: str) -> Optional[Dict]:
    """First fixture any of whose ``match`` substrings appears in ``headline``
    (case-insensitive); otherwise the empty-``match`` default, if present."""
    hl = headline.lower()
    default: Optional[Dict] = None
    for fx in fixtures:
        matches = [m.lower() for m in fx.get("match", [])]
        if not matches:
            default = default or fx
            continue
        if any(m in hl for m in matches):
            return fx
    return default


def _translate_fake(
    article: Article, *, on_step: Optional[StepCallback] = None
) -> TranslatorResult:
    """Deterministic offline translation from a JSON fixture.

    The fixture's ``payload`` is run through the *same* :func:`_validate_payload`
    path as the real providers, so malformed fixtures surface real
    ``TranslatorError``s exactly as a bad LLM response would. Fixtures are
    selected by matching the article headline.
    """
    _emit = on_step or (lambda *_: None)
    _emit("init", "Using fake translator (offline fixtures)…")
    fixture = _select_fixture(_load_fake_fixtures(), article.headline)
    if fixture is None:
        _emit("response", "no fixtures found; returning empty translation")
        return TranslatorResult(
            headline=article.headline, assignments=[], rationale="",
            model="fake:empty", provider="fake", raw_response="{}",
        )
    payload = fixture.get("payload", {})
    raw = json.dumps(payload)
    _emit("response", f"fixture '{fixture['_name']}' ({len(raw)} chars)")
    try:
        assignments, rationale = _validate_payload(payload)
    except TranslatorError as exc:
        exc.raw_response = raw  # type: ignore[attr-defined]
        raise
    _emit("validated", f"Validated {len(assignments)} assignment(s)")
    return TranslatorResult(
        headline=article.headline, assignments=assignments, rationale=rationale,
        model=f"fake:{fixture['_name']}", provider="fake", raw_response=raw,
    )


def fake_forced_by_env() -> bool:
    """True if ``TRANSLATOR_PROVIDER=fake`` forces the offline fake provider."""
    return os.environ.get("TRANSLATOR_PROVIDER", "").strip().lower() == "fake"


def _claude_output_format_enabled() -> bool:
    """Whether to pass ``output_format`` (schema-bound output) to the Claude CLI.

    Opt-in (default off): the Python SDK exposes the field, but the bundled
    ``claude`` CLI binary may reject it and fail with exit code 1. When off, the
    Claude path uses text + brace-matching + strict post-validation — the D4
    fallback, which still enforces A2's canonical shape via ``_validate_payload``.
    Enable with ``TRANSLATOR_CLAUDE_OUTPUT_FORMAT=1`` once the CLI supports it.
    """
    return os.environ.get("TRANSLATOR_CLAUDE_OUTPUT_FORMAT", "").strip().lower() in (
        "1", "true", "yes",
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def _apply_source_credibility(result: TranslatorResult, w: float) -> TranslatorResult:
    """Discount per-assignment likelihood ratios by source credibility ``w``.

    ``ε ← ε**w`` (Plan 2 §B1): ``w=1`` leaves ε unchanged; ``w=0`` flattens every
    ε to 1.0 (no information injected); ``w∈(0,1)`` interpolates. The best state
    stays pinned at 1.0 (``1.0**w == 1.0``), so the vector remains max-pinned.
    """
    if w >= 1.0:
        return result
    w = max(0.0, w)
    for a in result.assignments:
        a.state_probs = {s: (v ** w) for s, v in a.state_probs.items()}
    return result


def translate_article(
    article: Article,
    *,
    credibility: Optional[float] = None,
    provider: Optional[Provider] = None,
    client=None,
    on_step: Optional[StepCallback] = None,
) -> TranslatorResult:
    """Translate an :class:`Article` via the preferred (or requested) provider.

    ``credibility`` is the source-credibility weight ``w`` applied to the
    likelihood ratios (``ε ← ε**w``). When ``None`` it is looked up from the
    article's ``source_type`` (see :data:`SOURCE_TYPE_CREDIBILITY`).
    ``translate_headline`` passes ``credibility=1.0`` so a bare analyst-pasted
    headline is full-trust. Pass ``client=<openai-like>`` to force the OpenAI
    path (used by tests).
    """
    article.headline = article.headline.strip()
    if not article.headline:
        raise TranslatorError("Headline is empty.")

    w = (
        credibility if credibility is not None
        else SOURCE_TYPE_CREDIBILITY.get(article.source_type, 0.5)
    )

    if client is not None:
        return _apply_source_credibility(
            _translate_openai(article, client=client, on_step=on_step), w
        )

    # TRANSLATOR_PROVIDER forces a provider when the caller didn't specify one
    # (offline dev: TRANSLATOR_PROVIDER=fake). Explicit `provider=` wins.
    if provider is None:
        env_provider = os.environ.get("TRANSLATOR_PROVIDER", "").strip()
        if env_provider:
            provider = env_provider  # type: ignore[assignment]

    chosen: Optional[Provider]
    if provider is not None:
        chosen = provider
    else:
        providers = available_providers()
        chosen = providers[0] if providers else None

    if chosen == "fake":
        result = _translate_fake(article, on_step=on_step)
    elif chosen == "claude-code":
        result = _translate_claude_code(article, on_step=on_step)
    elif chosen == "openai":
        result = _translate_openai(article, on_step=on_step)
    else:
        raise TranslatorError(
            "No translator provider is available. Install Claude Code and sign in, "
            "export OPENAI_API_KEY, or set TRANSLATOR_PROVIDER=fake for offline fixtures."
        )
    return _apply_source_credibility(result, w)


def translate_headline(
    headline: str,
    *,
    provider: Optional[Provider] = None,
    client=None,
    on_step: Optional[StepCallback] = None,
) -> TranslatorResult:
    """Translate a bare headline (analyst paste) at full trust (``w=1.0``).

    Preserves the pre-B1a behaviour; for article-level input (body, source,
    source-type credibility) use :func:`translate_article`.
    """
    return translate_article(
        Article(headline=headline),
        credibility=1.0,
        provider=provider,
        client=client,
        on_step=on_step,
    )


__all__ = [
    "CLAUDE_DEFAULT_MODEL",
    "OPENAI_DEFAULT_MODEL",
    "SOURCE_TYPE_CREDIBILITY",
    "Article",
    "Provider",
    "SourceType",
    "TranslatorAssignment",
    "TranslatorError",
    "TranslatorResult",
    "available_providers",
    "fake_forced_by_env",
    "is_available",
    "translate_article",
    "translate_headline",
]
