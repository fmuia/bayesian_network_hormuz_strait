"""P4c — Network-view manual-override commit (AppTest).

Regression guard for the `state`-module name collision: the override panel loops
`for i, state in enumerate(states)`, which shadows a `state` module import — so
committing an override must not go through `state.<fn>`.
"""
from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _click(at, label):
    for b in at.button:
        if b.label == label:
            b.click()
            at.run()
            return True
    return False


def _app_with_node(node):
    at = AppTest.from_file("app/dashboard.py", default_timeout=90)
    at.session_state["use_fake_translator"] = True
    at.run()
    at.session_state["selected_node"] = node
    at.run()
    return at


def test_override_soft_commits_observation():
    at = _app_with_node("Tanker_Incidents")   # default sliders sum to 100 -> soft
    assert _click(at, "Set observation")
    assert not at.exception
    obs = at.session_state["observations"]
    assert len(obs) == 1 and obs[0]["soft_assignments"]


def test_override_hard_commits_observation():
    at = _app_with_node("Tanker_Incidents")
    for s, v in [("none", 100), ("isolated", 0), ("frequent", 0)]:
        at.session_state[f"soft_Tanker_Incidents_{s}"] = v
    at.run()
    assert _click(at, "Set observation")
    assert not at.exception
    obs = at.session_state["observations"]
    assert len(obs) == 1 and obs[0]["assignments"] == {"Tanker_Incidents": "none"}
