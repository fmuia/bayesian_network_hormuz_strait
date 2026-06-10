"""P9 — before/after delta chips on the scenario cards (AppTest)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import state  # noqa: E402

_TANKER = state.make_observation(
    day=1, headline="Fourth tanker incident this week", assignments={},
    soft_assignments={"Tanker_Incidents": {"none": 0.05, "isolated": 0.3, "frequent": 1.0}},
    item_id="t")


def _chips(at):
    md = " ".join(m.value for m in at.markdown)
    return re.findall(r"[▲▼•]\s*[-+−]?\d+\.\d+ pp", md)   # rendered chips, not the CSS rule


def _app():
    at = AppTest.from_file("app/dashboard.py", default_timeout=90)
    at.session_state["use_fake_translator"] = True
    at.run()
    return at


def test_no_delta_chips_before_any_observation():
    at = _app()
    assert _chips(at) == []          # nothing to compare against yet


def test_delta_chips_appear_after_an_observation():
    at = _app()
    at.session_state["observations"] = [_TANKER]
    at.run()
    assert not at.exception
    assert len(_chips(at)) == 3       # one per scenario (Stress / Prolonged / Severe)


def test_removing_the_observation_clears_the_chips():
    at = _app()
    at.session_state["observations"] = [_TANKER]
    at.run()
    assert len(_chips(at)) == 3
    at.session_state["observations"] = []   # remove / undo
    at.run()
    assert _chips(at) == []                  # restored cleanly
