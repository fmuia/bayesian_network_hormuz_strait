"""T06 — structured-reasoning pipeline tests (offline).

T06a: claim extraction — span-grounding + dedup (the testable core) and the
deterministic offline fake extractor.
"""
from __future__ import annotations

from src.translator import Article
from src.translator_pipeline import (
    Claim,
    _fake_claims,
    _parse_claims,
    article_text,
    extract_claims,
)

_TEXT = (
    "Iran seized a tanker near Hormuz. Oman is mediating back-channel talks. "
    "Insurers raised war-risk premia."
)


def _raw(span, **kw):
    base = {"subject": "", "predicate": "", "object": "", "confidence": 1.0}
    base.update(kw)
    base["verbatim_span"] = span
    return base


def test_parse_claims_keeps_grounded():
    claims = _parse_claims([_raw("Oman is mediating back-channel talks.")], _TEXT)
    assert len(claims) == 1
    assert isinstance(claims[0], Claim)
    assert claims[0].verbatim_span == "Oman is mediating back-channel talks."


def test_parse_claims_drops_ungrounded():
    """A span not present in the article (hallucinated/injected) is dropped."""
    raw = [
        _raw("Iran seized a tanker near Hormuz."),       # grounded -> kept
        _raw("A missile destroyed a refinery in Texas."),  # not in text -> dropped
    ]
    claims = _parse_claims(raw, _TEXT)
    assert [c.verbatim_span for c in claims] == ["Iran seized a tanker near Hormuz."]


def test_parse_claims_dedups_identical_span():
    raw = [
        _raw("Insurers raised war-risk premia."),
        _raw("Insurers raised  war-risk   premia."),  # same fact, whitespace differs
    ]
    claims = _parse_claims(raw, _TEXT)
    assert len(claims) == 1  # exact (whitespace-normalised) duplicate dropped


def test_parse_claims_grounding_is_whitespace_insensitive():
    # span with collapsed/extra whitespace still matches the article text
    claims = _parse_claims([_raw("Oman   is mediating back-channel talks.")], _TEXT)
    assert len(claims) == 1


def test_fake_claims_sentence_split_all_grounded():
    art = Article(headline="Gulf tension rises", body=_TEXT)
    raw = _fake_claims(art)
    assert len(raw) >= 3  # one per sentence (headline + 3 body sentences)
    text = article_text(art)
    for rc in raw:
        assert " ".join(rc["verbatim_span"].split()).lower() in " ".join(text.split()).lower()


def test_extract_claims_fake_offline():
    art = Article(headline="Gulf tension rises", body=_TEXT)
    claims = extract_claims(art, provider="fake")
    assert claims and all(isinstance(c, Claim) for c in claims)
    # every emitted claim is span-grounded in the article
    text_norm = " ".join(article_text(art).split()).lower()
    assert all(" ".join(c.verbatim_span.split()).lower() in text_norm for c in claims)


def test_extract_claims_empty_article():
    assert extract_claims(Article(headline="   "), provider="fake") == []
