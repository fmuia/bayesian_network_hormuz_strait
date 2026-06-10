"""P2 (Plan 5 A1a) — pure state helpers extracted to app/state.py.

These functions are import-safe without a Streamlit runtime; the st-coupled
wrappers are covered by tests/test_hitl.py via AppTest.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import state  # noqa: E402


def _obs(assignments=None, soft=None):
    return {"assignments": assignments or {}, "soft_assignments": soft or {}}


# ===== merged_evidence ordering invariants =================================


def test_merged_evidence_empty():
    assert state.merged_evidence([]) == ({}, {})


def test_hard_then_soft_then_hard_is_hard():
    obs = [
        _obs(assignments={"Tanker_Incidents": "frequent"}),
        _obs(soft={"Tanker_Incidents": {"none": 0.1, "isolated": 1.0, "frequent": 0.5}}),
        _obs(assignments={"Tanker_Incidents": "isolated"}),
    ]
    hard, soft = state.merged_evidence(obs)
    assert hard == {"Tanker_Incidents": "isolated"}
    assert "Tanker_Incidents" not in soft


def test_soft_then_hard_then_soft_is_soft():
    obs = [
        _obs(soft={"Tanker_Incidents": {"frequent": 1.0}}),
        _obs(assignments={"Tanker_Incidents": "none"}),
        _obs(soft={"Tanker_Incidents": {"isolated": 1.0}}),
    ]
    hard, soft = state.merged_evidence(obs)
    assert "Tanker_Incidents" not in hard
    assert soft["Tanker_Incidents"] == {"isolated": 1.0}


def test_two_nodes_both_preserved():
    obs = [
        _obs(soft={"Tanker_Incidents": {"frequent": 1.0}}),
        _obs(assignments={"Third_Party_Mediation": "active"}),
    ]
    hard, soft = state.merged_evidence(obs)
    assert hard == {"Third_Party_Mediation": "active"}
    assert set(soft) == {"Tanker_Incidents"}


def test_removal_recomputes_from_shorter_list():
    obs = [
        _obs(assignments={"Tanker_Incidents": "frequent"}),
        _obs(assignments={"Third_Party_Mediation": "active"}),
    ]
    hard, _ = state.merged_evidence(obs[:1])  # second observation removed
    assert hard == {"Tanker_Incidents": "frequent"}
    assert "Third_Party_Mediation" not in hard


def test_soft_values_coerced_to_float():
    _, soft = state.merged_evidence([_obs(soft={"Tanker_Incidents": {"none": 1, "isolated": 0}})])
    assert all(isinstance(v, float) for v in soft["Tanker_Incidents"].values())


# ===== P9 — before/after scenario deltas ===================================


def test_scenario_deltas_signed_pp_and_conserve():
    cur = {"A": 0.50, "B": 0.30, "C": 0.20}
    prev = {"A": 0.45, "B": 0.35, "C": 0.20}
    d = state.scenario_deltas(cur, prev)
    assert abs(d["A"] - 5.0) < 1e-9        # +5 pp
    assert abs(d["B"] + 5.0) < 1e-9        # -5 pp
    assert abs(d["C"]) < 1e-9              # unchanged
    assert abs(sum(d.values())) < 1e-9     # probability conserved -> deltas sum to 0


def test_scenario_deltas_missing_prev_is_zero():
    assert state.scenario_deltas({"A": 0.6}, {})["A"] == 0.0   # no prior -> no delta


# ===== observation + review-queue helpers ==================================


def _result():
    a = SimpleNamespace(node="Tanker_Incidents", state="frequent", reason="strikes",
                        state_probs={"none": 0.05, "isolated": 0.3, "frequent": 1.0})
    return SimpleNamespace(headline="A tanker was struck near Hormuz.", relevance="yes",
                           model="m", provider="fake", rationale="r", assignments=[a])


def test_make_observation_record():
    rec = state.make_observation(day=2, headline="h", assignments={"X": "y"}, item_id="id1")
    assert rec["id"] == "id1" and rec["day"] == 2 and rec["headline"] == "h"
    assert rec["assignments"] == {"X": "y"}


def test_make_review_item_shape():
    item = state.make_review_item(_result(), day=3, item_id="abc")
    assert item["id"] == "abc" and item["day"] == 3 and item["relevance"] == "yes"
    assert item["assignments"][0]["node"] == "Tanker_Incidents"
    assert item["assignments"][0]["state_probs"]["frequent"] == 1.0


def test_review_item_soft_evidence_passthrough():
    item = state.make_review_item(_result(), day=1, item_id="x")
    soft, reasons = state.review_item_soft_evidence(item)
    assert soft["Tanker_Incidents"]["frequent"] == 1.0
    assert reasons["Tanker_Incidents"] == "strikes"


def test_review_item_soft_evidence_override():
    item = state.make_review_item(_result(), day=1, item_id="x")
    soft, reasons = state.review_item_soft_evidence(
        item, state_overrides={"Tanker_Incidents": "isolated"})
    assert soft["Tanker_Incidents"]["isolated"] == 1.0
    assert soft["Tanker_Incidents"]["frequent"] == state.OVERRIDE_FLOOR
    assert "analyst-edited" in reasons["Tanker_Incidents"]


def test_remove_from_review():
    q = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    assert [x["id"] for x in state.remove_from_review(q, "b")] == ["a", "c"]


# ===== named-session persistence (round-trip) ==============================


def test_load_missing_store_returns_empty(tmp_path):
    assert state.load_session_store(path=tmp_path / "nope.json") == {}


def test_session_save_load_roundtrip(tmp_path):
    store_path = tmp_path / "sessions.json"
    snap_state = {
        "current_day": 4,
        "observations": [state.make_observation(
            day=1, headline="h", assignments={"X": "y"}, item_id="i")],
        "last_translation": {"foo": "bar"},
        "translator_error": None,
        "translator_raw": "raw",
        "selected_node": "Tanker_Incidents",
    }
    snap = state.session_snapshot(snap_state, saved_at="2026-01-01T00:00:00Z")
    state.write_session_store({"mysession": snap}, path=store_path)

    payload = state.load_session_store(path=store_path)["mysession"]
    restored = state.restore_values(payload)
    assert restored["current_day"] == 4
    assert restored["observations"] == snap_state["observations"]
    assert restored["selected_node"] == "Tanker_Incidents"
    # keys absent from the snapshot fall back to a *fresh copy* of the default,
    # never an alias of the shared module-level SS_DEFAULTS (the P2 aliasing bug).
    assert restored["review_queue"] == []
    assert restored["review_queue"] is not state.SS_DEFAULTS["review_queue"]
