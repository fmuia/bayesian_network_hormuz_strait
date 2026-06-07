"""Translator tests.

T00: the offline `fake` provider scaffolding — deterministic, no network — that
both the test suite and the dashboard use for offline play. Later cards (T01+)
extend this file.
"""
from __future__ import annotations

import pytest

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from src.inference import BNInferenceEngine
from src.network import STATES, build_network
from src.translator import (
    TranslatorError,
    TranslatorResult,
    _system_prompt,
    _validate_payload,
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


# ===== T01 — A1 likelihood semantics =======================================


def _payload(node: str, eps: dict) -> dict:
    return {
        "assignments": [
            {
                "node": node,
                "state": max(eps, key=eps.get),
                "reason": "test",
                "state_probs": [{"state": s, "prob": v} for s, v in eps.items()],
            }
        ],
        "overall_rationale": "r",
    }


def test_likelihood_combines_with_prior_once():
    """§A1 minimal example: a likelihood ratio multiplies the prior exactly once.

    prior (0.9, 0.1), injected ε (0.8, 0.2) -> posterior ∝ (0.72, 0.02)
    = (0.973, 0.027). The single-multiplication property A1's ε output gives.
    """
    net = DiscreteBayesianNetwork()
    net.add_node("X")
    net.add_cpds(
        TabularCPD("X", 2, [[0.9], [0.1]], state_names={"X": ["a", "b"]})
    )
    net.check_model()
    ve = VariableElimination(net)
    eps = TabularCPD("X", 2, [[0.8], [0.2]], state_names={"X": ["a", "b"]})
    post = ve.query(["X"], virtual_evidence=[eps], show_progress=False)
    vals = np.array([post.get_value(X="a"), post.get_value(X="b")])
    assert np.allclose(vals, [0.973, 0.027], atol=1e-3)


def test_soft_evidence_single_multiplication_on_engine():
    """Through our own stack: posterior == normalise(prior * ε) for the node."""
    eng = BNInferenceEngine(build_network("latent_regime"))
    node = "Tanker_Incidents"
    prior = eng.get_node_marginal(node)
    eps = {"none": 1.0, "isolated": 0.5, "frequent": 0.1}
    eng.update_soft_evidence({node: eps})
    post = eng.get_node_marginal(node)
    raw = {s: prior[s] * eps[s] for s in STATES[node]}
    z = sum(raw.values())
    for s in STATES[node]:
        assert abs(post[s] - raw[s] / z) < 1e-6


def test_validator_accepts_maxpinned():
    asg, _ = _validate_payload(
        _payload("Tanker_Incidents", {"none": 0.1, "isolated": 0.4, "frequent": 1.0})
    )
    assert asg[0].state == "frequent"
    assert asg[0].state_probs["frequent"] == 1.0
    # not renormalised away from the ε scale
    assert asg[0].state_probs["isolated"] == 0.4


def test_validator_rejects_not_maxpinned():
    with pytest.raises(TranslatorError):
        _validate_payload(
            _payload("Tanker_Incidents", {"none": 0.3, "isolated": 0.5, "frequent": 0.2})
        )


def test_validator_rejects_zero_likelihood():
    with pytest.raises(TranslatorError):
        _validate_payload(
            _payload("Tanker_Incidents", {"none": 0.0, "isolated": 0.5, "frequent": 1.0})
        )


def test_validator_floors_unmentioned_state():
    """A state the model omits is floored (not zeroed) so its posterior survives."""
    payload = {
        "assignments": [
            {
                "node": "Tanker_Incidents",
                "state": "frequent",
                "reason": "only two states mentioned",
                "state_probs": [
                    {"state": "frequent", "prob": 1.0},
                    {"state": "isolated", "prob": 0.3},
                ],
            }
        ],
        "overall_rationale": "r",
    }
    asg, _ = _validate_payload(payload)
    assert asg[0].state_probs["none"] > 0.0  # floored, not 0


def test_prompt_drops_sum_to_one_instruction():
    p = " ".join(_system_prompt().lower().split())  # collapse line-wrap whitespace
    assert "must sum to 1" not in p          # the old affirmative instruction is gone
    assert "relative likelihood" in p         # new likelihood-ratio instruction
    assert "must not sum to 1" in p
