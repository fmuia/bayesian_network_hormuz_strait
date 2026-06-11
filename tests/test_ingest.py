"""URL / title ingestion (post-P5 improvements) — pure, offline unit tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from src import ingest

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "articles"


# --- classification --------------------------------------------------------

@pytest.mark.parametrize("text,kind", [
    ("https://www.reuters.com/world/x", "url"),
    ("http://bbc.co.uk/news", "url"),
    ("reuters.com/world/middle-east/story", "url"),   # bare host + path
    ("www.aljazeera.com", "url"),
    ("Iran suspends Hormuz traffic inspections", "title"),   # has spaces
    ("Hormuz", "title"),                              # one word, no dot
    ("", "title"),
])
def test_classify_input(text, kind):
    assert ingest.classify_input(text) == kind


# --- site identification ---------------------------------------------------

def test_identify_site_known_and_normalised():
    assert ingest.identify_site("https://www.reuters.com/x").name == "Reuters"
    assert ingest.identify_site("https://edition.cnn.com/2026/x").name == "CNN"  # subdomain
    assert ingest.identify_site("http://bbc.co.uk/news").source_type == "commercial_press"
    assert ingest.identify_site("https://presstv.ir/x").source_type == "state_media"


def test_identify_site_unknown_is_none():
    assert ingest.identify_site("https://example.com/story") is None
    assert ingest.identify_site("https://notreuters.com/x") is None   # not a suffix match


# --- extraction ------------------------------------------------------------

def test_extract_article_from_fixture():
    html = (_FIXTURES / "reuters.html").read_text(encoding="utf-8")
    headline, body = ingest.extract_article(html)
    assert "Tanker struck in the Strait of Hormuz" in headline   # og:title
    assert "war-risk premiums" in body                           # JSON-LD articleBody


def test_extract_article_h1_fallback_when_no_meta():
    html = "<html><body><h1>Strait closure feared</h1><p>Body text here.</p></body></html>"
    headline, body = ingest.extract_article(html)
    assert headline == "Strait closure feared"
    assert "Body text here." in body


def test_extract_article_no_headline_returns_empty():
    headline, _ = ingest.extract_article("<html><body><p>just a paragraph</p></body></html>")
    assert headline == ""


# --- orchestrator: the four outcomes ---------------------------------------

def test_ingest_manual_body_forces_headline():
    res = ingest.ingest("Tanker hit near Hormuz", "Fuller body text.")
    assert res.kind == "manual" and res.ok
    assert res.headline == "Tanker hit near Hormuz" and res.body == "Fuller body text."


def test_ingest_bare_title():
    res = ingest.ingest("Iran suspends Hormuz traffic inspections")
    assert res.kind == "title" and res.ok
    assert res.headline == "Iran suspends Hormuz traffic inspections" and res.body == ""


def test_ingest_known_url_fetches_and_splits():
    res = ingest.ingest("https://www.reuters.com/world/x", fetcher=ingest.fake_fetcher)
    assert res.ok and res.kind == "url"
    assert "Tanker struck in the Strait of Hormuz" in res.headline
    assert "war-risk premiums" in res.body
    assert res.source == "Reuters" and res.source_type == "wire_service"
    assert res.site_name == "Reuters"


def test_ingest_unlisted_url_is_rejected():
    res = ingest.ingest("https://example.com/some-story")
    assert not res.ok and res.kind == "url"
    assert "example.com" in res.message and "supported outlet" in res.message


def test_ingest_known_url_empty_extraction_is_rejected():
    res = ingest.ingest(
        "https://www.reuters.com/x",
        fetcher=lambda _u: "<html><body><p>no headline here</p></body></html>",
    )
    assert not res.ok and "couldn’t find an article headline" in res.message


def test_ingest_fetch_failure_degrades_gracefully():
    def _boom(_url):
        raise RuntimeError("network down")
    res = ingest.ingest("https://www.reuters.com/x", fetcher=_boom)
    assert not res.ok and "Couldn’t read the article" in res.message
