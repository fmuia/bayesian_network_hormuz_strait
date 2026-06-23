"""Keyword matcher for the offline fake translator (quadruple-check regression).

The fake keyword fallback matched bare substrings, so 'port' fired inside
'report' — a calm headline mapped to the worst states. The matcher now requires
a word start while still allowing suffixes ('lead time' ⊂ 'lead times')."""
from __future__ import annotations

from src.translator import _keyword_hit


def test_substring_inside_a_word_does_not_match():
    assert not _keyword_hit("port", "suppliers report stable conditions")
    assert not _keyword_hit("oil", "the plan was spoiled")


def test_word_start_matches_including_suffix():
    assert _keyword_hit("port", "the port is congested")
    assert _keyword_hit("lead time", "magnet lead times stretch to 14 weeks")
    assert _keyword_hit("force majeure", "three suppliers declare force majeure")


def test_non_alpha_keyword_matches():
    assert _keyword_hit("%", "prices jump 35% this week")
