"""Dashboard session state — defaults, evidence merging, named-session
persistence, and the HITL review-queue helpers (Plan 5 P2 / A1a).

Extracted from ``app/dashboard.py``. The data-transforming functions
(``merged_evidence``, ``make_observation``, ``make_review_item``,
``review_item_soft_evidence``, ``remove_from_review``, the session-store I/O,
``session_snapshot`` and ``restore_values``) are **pure** and import-safe
without a Streamlit runtime — they are unit-tested in ``tests/test_state.py``.
The thin ``st.session_state``-coupled wrappers at the bottom are exercised
through ``AppTest``.
"""
from __future__ import annotations

import copy
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

from src.evidence import Observation
from src.scenario import STATES

ROOT = Path(__file__).resolve().parents[1]

# ε floor for the non-chosen states when an analyst edits a state in Triage.
OVERRIDE_FLOOR = 0.01

SS_DEFAULTS: Dict[str, object] = {
    "observations": [],
    "current_day": 1,
    "last_translation": None,
    "translator_error": None,
    "translator_raw": "",
    "pending_article": None,
    "selected_node": None,
    "review_queue": [],          # T12: translations awaiting analyst review
    "locked_spec_json": "",       # Plan 4 elicitation layer: locked elicited network
    "current_run_dict": None,     # the elicitation run being inspected
}

# The slice of session state persisted by a named session save.
SNAPSHOT_KEYS = (
    "current_day", "observations", "last_translation",
    "translator_error", "translator_raw", "selected_node",
)

DEFAULT_STORE_PATH = ROOT / "data" / "dashboard_saved_sessions.json"


# ===========================================================================
# Pure logic (no Streamlit runtime needed)
# ===========================================================================


def merged_evidence(
    observations: List[dict],
) -> Tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
    """Merge a list of observations into (hard, soft) evidence.

    Latest observation wins on conflict, in insertion order; a hard assignment
    clears any prior soft evidence on that node and vice-versa.
    """
    hard_merged: Dict[str, str] = {}
    soft_merged: Dict[str, Dict[str, float]] = {}
    for obs in observations:
        for node, state in obs.get("assignments", {}).items():
            hard_merged[node] = state
            soft_merged.pop(node, None)
        for node, dist in obs.get("soft_assignments", {}).items():
            soft_merged[node] = {k: float(v) for k, v in dist.items()}
            hard_merged.pop(node, None)
    return hard_merged, soft_merged


def scenario_deltas(
    current_means: Dict[str, float],
    prev_means: Dict[str, float],
) -> Dict[str, float]:
    """Signed percentage-point change per scenario, ``(current - prev) * 100``
    (Plan 5 P9 / C7 / V9) — the effect of the most recent observation. A scenario
    missing from ``prev_means`` is treated as unchanged."""
    return {
        s: (current_means[s] - prev_means.get(s, current_means[s])) * 100.0
        for s in current_means
    }


def override_to_observation(vals: Dict[str, int]):
    """Raw manual-override slider values (state -> 0..100, any positive sum) into
    either a hard pin or a sum-normalised soft distribution (Plan 5 follow-up, V4).

    Returns ``(pinned_state | None, soft_dist | None)``:
    - exactly one non-zero state -> ``(state, None)`` (a hard observation),
    - two or more non-zero states -> ``(None, {state: v / total})`` (normalised soft),
    - all zero -> ``(None, None)`` (nothing to apply).

    Auto-normalising by the total means the user no longer has to make the sliders
    sum to exactly 100.
    """
    nonzero = {s: v for s, v in vals.items() if v > 0}
    total = sum(nonzero.values())
    if total <= 0:
        return None, None
    if len(nonzero) == 1:
        return next(iter(nonzero)), None
    return None, {s: v / total for s, v in vals.items()}


def make_observation(
    *,
    day: int,
    headline: str,
    assignments: Dict[str, str],
    soft_assignments: Optional[Dict[str, Dict[str, float]]] = None,
    rationale: str = "",
    per_assignment_reasons: Optional[Dict[str, str]] = None,
    source: str = "translator",
    item_id: Optional[str] = None,
) -> dict:
    """Build a JSON-friendly observation record (with an id)."""
    obs = Observation(
        day=day,
        headline=headline,
        assignments=dict(assignments),
        soft_assignments=dict(soft_assignments or {}),
        rationale=rationale,
        per_assignment_reasons=per_assignment_reasons or {},
        source=source,
    )
    return {"id": item_id or uuid.uuid4().hex, **asdict(obs)}


def make_review_item(result, *, day: int, item_id: Optional[str] = None) -> dict:
    """Normalise a TranslatorResult into a review-queue entry (JSON-friendly)."""
    return {
        "id": item_id or uuid.uuid4().hex,
        "headline": result.headline,
        "day": day,
        "relevance": result.relevance,
        "model": result.model,
        "provider": result.provider,
        "rationale": result.rationale,
        "assignments": [
            {"node": a.node, "state": a.state,
             "state_probs": dict(a.state_probs), "reason": a.reason}
            for a in result.assignments
        ],
    }


def review_item_soft_evidence(
    item: dict,
    *,
    state_overrides: Optional[Dict[str, str]] = None,
    override_floor: float = OVERRIDE_FLOOR,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    """(soft_assignments, per_assignment_reasons) for a review item.

    An analyst state edit becomes confident soft evidence on the chosen state.
    """
    overrides = state_overrides or {}
    soft: Dict[str, Dict[str, float]] = {}
    reasons: Dict[str, str] = {}
    for a in item["assignments"]:
        node = a["node"]
        chosen = overrides.get(node, a["state"])
        if node in overrides and chosen != a["state"]:
            soft[node] = {s: (1.0 if s == chosen else override_floor) for s in STATES[node]}
            reasons[node] = f"{a['reason']} (analyst-edited: {a['state']} → {chosen})"
        else:
            soft[node] = dict(a["state_probs"])
            reasons[node] = a["reason"]
    return soft, reasons


def remove_from_review(queue: List[dict], item_id: str) -> List[dict]:
    """Return the review queue without the entry whose id is ``item_id``."""
    return [x for x in queue if x["id"] != item_id]


def session_snapshot(state: Dict[str, object], *, saved_at: str) -> dict:
    """Serialise the persisted slice of session state (``state`` is a plain dict)."""
    snap: Dict[str, object] = {"saved_at": saved_at}
    for k in SNAPSHOT_KEYS:
        snap[k] = state[k]
    return snap


def restore_values(payload: dict) -> Dict[str, object]:
    """Map a saved payload onto the full default key set (missing -> default).

    Missing keys fall back to a *copy* of the default so the shared module-level
    ``SS_DEFAULTS`` (mutable lists) is never aliased into session state.
    """
    return {k: payload.get(k, copy.deepcopy(SS_DEFAULTS[k])) for k in SS_DEFAULTS}


def load_session_store(path: Path = DEFAULT_STORE_PATH) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_session_store(store: Dict[str, Dict], path: Path = DEFAULT_STORE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2, sort_keys=True), encoding="utf-8")


# ===========================================================================
# Streamlit-coupled wrappers (read/write st.session_state)
# ===========================================================================


def init_session_state() -> None:
    # deepcopy so the shared module-level SS_DEFAULTS (mutable lists) is never
    # aliased into session state — otherwise appends would leak across reruns.
    for key, value in SS_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(value)


def record_observation(
    headline: str,
    assignments: Dict[str, str],
    soft_assignments: Optional[Dict[str, Dict[str, float]]] = None,
    rationale: str = "",
    per_assignment_reasons: Optional[Dict[str, str]] = None,
    source: str = "translator",
) -> None:
    st.session_state.observations.append(make_observation(
        day=st.session_state.current_day,
        headline=headline,
        assignments=assignments,
        soft_assignments=soft_assignments,
        rationale=rationale,
        per_assignment_reasons=per_assignment_reasons,
        source=source,
    ))


def current_evidence() -> Tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
    return merged_evidence(st.session_state.observations)


def build_review_item(result) -> dict:
    return make_review_item(result, day=st.session_state.current_day)


def inject_review_item(item: dict, *, state_overrides: Optional[dict] = None) -> None:
    soft, reasons = review_item_soft_evidence(item, state_overrides=state_overrides)
    record_observation(
        headline=item["headline"], assignments={}, soft_assignments=soft,
        rationale=item["rationale"], per_assignment_reasons=reasons, source="translator",
    )


def remove_review_item(item_id: str) -> None:
    st.session_state.review_queue = remove_from_review(
        st.session_state.review_queue, item_id,
    )


def save_named_session(name: str) -> None:
    store = load_session_store()
    snap_state = {k: st.session_state[k] for k in SNAPSHOT_KEYS}
    store[name] = session_snapshot(
        snap_state, saved_at=datetime.now(timezone.utc).isoformat(),
    )
    write_session_store(store)


def restore_named_session(name: str) -> bool:
    store = load_session_store()
    payload = store.get(name)
    if payload is None:
        return False
    for key, value in restore_values(payload).items():
        st.session_state[key] = value
    return True


def delete_named_session(name: str) -> bool:
    store = load_session_store()
    if name not in store:
        return False
    del store[name]
    write_session_store(store)
    return True
