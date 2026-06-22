"""T12 — in-session HITL review queue (driven through the app via AppTest).

Offline (fake provider). Verifies that flagged translations are held in the
review queue and do NOT inject until approved; approve injects, reject discards,
and 'require review before inject' holds even clearly-relevant translations.
"""
from __future__ import annotations

from streamlit.testing.v1 import AppTest

_TRIAGE = "⚖️  Triage"  # must match _VIEW_TRIAGE in dashboard.py
_PARTIAL = {  # fake partial_ambiguous fixture -> relevance "partial"
    "raw": "Brent crude prices climb on Gulf jitters",
    "body": "", "source": "", "source_type_label": "(unspecified — full trust)",
}
_RELEVANT = {  # fake tanker fixture -> relevance "yes"
    "raw": "Tanker struck in the Strait of Hormuz",
    "body": "", "source": "", "source_type_label": "(unspecified — full trust)",
}


def _app():
    at = AppTest.from_file("app/dashboard.py", default_timeout=90)
    at.session_state["use_fake_translator"] = True
    at.run()
    return at


def _click(at, label):
    for b in at.button:
        if b.label == label:
            b.click()
            at.run()
            return True
    return False


def test_partial_held_then_approved_injects():
    at = _app()
    at.session_state["pending_article"] = _PARTIAL
    at.run()
    # held, not injected
    assert len(at.session_state["review_queue"]) == 1
    assert len(at.session_state["observations"]) == 0
    # approve from the Triage view
    at.session_state["active_view"] = _TRIAGE
    at.run()
    assert _click(at, "✓ Approve")
    assert len(at.session_state["observations"]) == 1
    assert len(at.session_state["review_queue"]) == 0


def test_reject_discards_without_injecting():
    at = _app()
    at.session_state["pending_article"] = _PARTIAL
    at.run()
    at.session_state["active_view"] = _TRIAGE
    at.run()
    assert _click(at, "✕ Reject")
    assert len(at.session_state["observations"]) == 0
    assert len(at.session_state["review_queue"]) == 0


def test_review_before_inject_holds_relevant():
    at = _app()
    at.session_state["review_before_inject"] = True
    at.run()
    at.session_state["pending_article"] = _RELEVANT  # relevance "yes"
    at.run()
    assert len(at.session_state["review_queue"]) == 1   # held despite being relevant
    assert len(at.session_state["observations"]) == 0


def test_relevant_auto_injects_when_review_off():
    at = _app()  # review_before_inject defaults off
    at.session_state["pending_article"] = _RELEVANT
    at.run()
    assert len(at.session_state["review_queue"]) == 0
    assert len(at.session_state["observations"]) == 1
