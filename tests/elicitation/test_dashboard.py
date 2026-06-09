"""Smoke test: the main dashboard renders end-to-end with the elicitation layer.

Confirms the integration (locked-aware engine, per-CPT kappa in the credible
intervals, and the elicitation panel) loads without exception in the default
bootstrap state. A real elicitation run needs LLMs and is not triggered here.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_with_elicitation_layer() -> None:
    at = AppTest.from_file("app/dashboard.py", default_timeout=200)
    at.run()
    assert not at.exception
    assert any("Elicitation layer" in str(e.label) for e in at.expander)
    # defaults to the bootstrap CPTs (no elicitation locked)
    assert at.session_state["locked_spec_json"] == ""
