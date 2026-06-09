"""P1 (Plan 5 A2 / V8) — CSS extracted to app/styles.css and injected at startup.

Guards the extraction: the stylesheet is a fully-resolved static file (no leftover
f-string tokens), the dashboard no longer inlines the rule block, and it injects
the stylesheet on boot.
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_styles_css_exists_and_is_fully_resolved():
    css = (ROOT / "app" / "styles.css").read_text(encoding="utf-8")
    assert css.strip()
    # no f-string artefacts leaked from the extraction
    assert "{{" not in css and "}}" not in css
    for tok in ("{NAVY}", "{TEAL}", "{PANEL}", "{GREEN}", "{RED}", "{RULE}", "{MUTED}"):
        assert tok not in css
    assert css.count("{") == css.count("}")   # balanced braces
    assert "#1B2A3D" in css                    # NAVY resolved to a literal hex


def test_dashboard_has_no_inline_css_rule_block():
    src = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
    assert "_inject_styles" in src
    assert src.count("<style>") == 1           # only the loader wrapper, not the rules


def test_dashboard_injects_stylesheet_on_boot():
    at = AppTest.from_file("app/dashboard.py", default_timeout=90)
    at.session_state["use_fake_translator"] = True
    at.run()
    assert not at.exception
    blob = " ".join(m.value for m in at.markdown)
    assert "<style>" in blob and "stSidebarCollapseButton" in blob
