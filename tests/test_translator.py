"""Translator tests.

T00: the offline `fake` provider scaffolding — deterministic, no network — that
both the test suite and the dashboard use for offline play. Later cards (T01+)
extend this file.
"""
from __future__ import annotations

import pytest

from src.translator import (
    TranslatorError,
    TranslatorResult,
    available_providers,
    fake_forced_by_env,
    translate_headline,
)


def test_fake_provider_returns_keyed_fixture():
    """A headline matching a fixture's `match` substrings selects that fixture."""
    res = translate_headline("Tanker struck in the Strait of Hormuz", provider="fake")
    assert isinstance(res, TranslatorResult)
    assert res.provider == "fake"
    assert any(a.node == "Tanker_Incidents" for a in res.assignments)


def test_fake_provider_falls_back_to_default():
    """An unmatched headline still yields a deterministic (default-fixture) result."""
    res = translate_headline(
        "An unrelated sentence about nothing in particular", provider="fake"
    )
    assert res.provider == "fake"
    assert res.assignments  # default fixture produces at least one assignment


def test_fake_provider_malformed_fixture_raises():
    """Malformed fixtures flow through the real validator and raise loudly."""
    with pytest.raises(TranslatorError):
        translate_headline("This headline is malformed on purpose", provider="fake")
    with pytest.raises(TranslatorError):
        translate_headline("A headline with a badstate token", provider="fake")


def test_fake_provider_via_env(monkeypatch):
    """TRANSLATOR_PROVIDER=fake forces the fake provider when none is passed."""
    monkeypatch.setenv("TRANSLATOR_PROVIDER", "fake")
    assert fake_forced_by_env() is True
    res = translate_headline("Anything at all")
    assert res.provider == "fake"


def test_fake_not_auto_selected():
    """The fake provider is never advertised by available_providers()."""
    assert "fake" not in available_providers()
