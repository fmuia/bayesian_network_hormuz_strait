"""T06 — structured-reasoning pipeline tests (offline).

T06a: claim extraction — span-grounding + dedup (the testable core) and the
deterministic offline fake extractor.
"""
from __future__ import annotations

import pytest

from src.translator import Article, TranslatorError
from src.translator_pipeline import (
    Claim,
    ClaimMapping,
    _fake_claims,
    _parse_claims,
    _parse_mappings,
    article_text,
    extract_claims,
    map_claims,
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


# ===== T06b — per-claim node mapping =======================================


def _claim(span):
    return Claim(subject="", predicate="", object="", verbatim_span=span)


def test_parse_mappings_valid():
    claims = [_claim("A tanker was struck near Hormuz.")]
    raw = [{
        "claim_index": 0, "node": "Tanker_Incidents", "state": "frequent",
        "reason": "repeated strikes",
        "state_probs": [{"state": "none", "value": 0.05},
                        {"state": "isolated", "value": 0.3},
                        {"state": "frequent", "value": 1.0}],
    }]
    maps = _parse_mappings(raw, claims)
    assert len(maps) == 1
    assert isinstance(maps[0], ClaimMapping)
    assert maps[0].node == "Tanker_Incidents"
    assert maps[0].state == "frequent"
    assert maps[0].supporting_span == "A tanker was struck near Hormuz."
    assert maps[0].state_probs["frequent"] == 1.0


def test_parse_mappings_unmapped_dropped():
    """A claim mapped to no node (node == '') yields no assignment."""
    claims = [_claim("Markets were quiet.")]
    raw = [{"claim_index": 0, "node": "", "state": "", "state_probs": [], "reason": ""}]
    assert _parse_mappings(raw, claims) == []


def test_parse_mappings_unknown_node_raises():
    """Out-of-snapshot node is rejected (A2 discipline)."""
    claims = [_claim("x")]
    raw = [{"claim_index": 0, "node": "Nope_Node", "state": "frequent",
            "state_probs": [{"state": "frequent", "value": 1.0}], "reason": ""}]
    with pytest.raises(TranslatorError):
        _parse_mappings(raw, claims)


def test_parse_mappings_bad_eps_raises():
    """Non-max-pinned ε is rejected (A1 discipline, reused validator)."""
    claims = [_claim("A tanker was struck near Hormuz.")]
    raw = [{"claim_index": 0, "node": "Tanker_Incidents", "state": "frequent", "reason": "",
            "state_probs": [{"state": "none", "value": 0.3},
                            {"state": "isolated", "value": 0.5},
                            {"state": "frequent", "value": 0.2}]}]  # no 1.0
    with pytest.raises(TranslatorError):
        _parse_mappings(raw, claims)


def test_map_claims_fake_offline():
    art = Article(headline="Gulf tension",
                  body="A tanker was struck near Hormuz. "
                       "Oman is mediating between Washington and Tehran.")
    claims = extract_claims(art, provider="fake")
    maps = map_claims(art, claims, provider="fake")
    nodes = {m.node for m in maps}
    assert "Tanker_Incidents" in nodes          # 'tanker' keyword
    assert "Third_Party_Mediation" in nodes     # 'mediat'/'oman' keyword
    # every mapping is max-pinned (A1) and carries its supporting span
    for m in maps:
        assert abs(max(m.state_probs.values()) - 1.0) < 1e-9
        assert m.supporting_span
