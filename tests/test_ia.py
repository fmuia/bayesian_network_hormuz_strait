"""P5 (Plan 5 A5) — information-architecture invariants for the nav.

The dashboard's top-level views register through one ``st.segmented_control``;
each gets a distinct icon, and 🧪 is reserved for the Elicitation lab (an
expander, not a view).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _view_labels():
    src = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
    return re.findall(r'_VIEW_\w+ = "([^"]+)"', src)


def test_five_nav_views():
    assert len(_view_labels()) == 5


def test_nav_view_icons_are_distinct():
    icons = [label.split()[0] for label in _view_labels()]   # leading emoji token
    assert len(set(icons)) == len(icons)


def test_triage_does_not_collide_with_elicitation_lab():
    # 🧪 belongs to the elicitation expander; no nav view may use it.
    assert all(label.split()[0] != "🧪" for label in _view_labels())
