"""URL ingestion in the translator sidebar (AppTest, offline via fixtures).

In fake mode the ingest layer fetches saved HTML from tests/fixtures/articles/
instead of the network, so these run with no internet.
"""
from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _app():
    at = AppTest.from_file("app/dashboard.py", default_timeout=90)
    at.session_state["use_fake_translator"] = True
    at.run()
    return at


def _submit(at, raw, body=""):
    at.session_state["pending_article"] = {
        "raw": raw, "body": body, "source": "",
        "source_type_label": "(unspecified — full trust)"}
    at.run()


def test_known_news_url_fetches_splits_and_injects():
    at = _app()
    _submit(at, "https://www.reuters.com/world/middle-east/tanker-hormuz")
    assert not at.exception
    assert len(at.session_state["observations"]) == 1     # auto-injected
    lt = at.session_state["last_translation"]
    assert lt and "Strait of Hormuz" in lt["headline"]    # headline split from the page


def test_unlisted_url_is_rejected_and_injects_nothing():
    at = _app()
    _submit(at, "https://example.com/some-story")
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "Rejected" in md and "example.com" in md       # clear rejection message
    assert len(at.session_state["observations"]) == 0     # nothing injected
    assert at.session_state["last_translation"] is None


def test_bare_title_still_translates():
    at = _app()
    _submit(at, "Tanker struck in the Strait of Hormuz")
    assert not at.exception
    assert len(at.session_state["observations"]) == 1


def test_example_button_translates_its_headline_not_empty():
    """Regression: the Examples buttons must feed the headline through the `raw`
    key (ingest reads `raw`, not the old `headline`), else they translate an
    empty headline."""
    from src.evidence import EXAMPLE_HEADLINES

    at = _app()
    ex = EXAMPLE_HEADLINES[0]
    at.get("button")  # ensure widgets are realised
    for b in at.button:
        if b.label == ex.text:
            b.click().run()
            break
    else:
        raise AssertionError(f"example button {ex.text!r} not found")
    assert not at.exception
    lt = at.session_state["last_translation"]
    assert lt is not None and lt["headline"] == ex.text   # the example text, not ""
