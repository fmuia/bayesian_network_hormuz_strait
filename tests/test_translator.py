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
    SOURCE_TYPE_CREDIBILITY,
    Article,
    TranslatorError,
    TranslatorResult,
    _extract_json_block,
    _finalize_payload,
    _states_hash,
    _system_prompt,
    _validate_payload,
    available_providers,
    fake_forced_by_env,
    translate_article,
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
                "state_probs": [{"state": s, "value": v} for s, v in eps.items()],
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
                    {"state": "frequent", "value": 1.0},
                    {"state": "isolated", "value": 0.3},
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


# ===== T02 — A2 schema hardening ===========================================


def _value_item(node, eps):
    """Canonical {state, value} payload."""
    return {
        "assignments": [
            {
                "node": node,
                "state": max(eps, key=eps.get),
                "reason": "t",
                "state_probs": [{"state": s, "value": v} for s, v in eps.items()],
            }
        ],
        "overall_rationale": "r",
    }


def test_validator_rejects_dict_shape():
    """The old permissive dict shape (C6) is no longer accepted."""
    bad = {
        "assignments": [
            {
                "node": "Tanker_Incidents",
                "state": "frequent",
                "reason": "t",
                "state_probs": {"none": 0.05, "isolated": 0.3, "frequent": 1.0},
            }
        ],
        "overall_rationale": "r",
    }
    with pytest.raises(TranslatorError):
        _validate_payload(bad)


def test_validator_rejects_json_string_shape():
    """The old JSON-encoded-string shape (C6) is no longer accepted."""
    bad = {
        "assignments": [
            {
                "node": "Tanker_Incidents",
                "state": "frequent",
                "reason": "t",
                "state_probs": '[{"state": "frequent", "value": 1.0}]',
            }
        ],
        "overall_rationale": "r",
    }
    with pytest.raises(TranslatorError):
        _validate_payload(bad)


def test_validator_rejects_legacy_prob_key():
    """An item using the old `prob` key (no `value`) is rejected, not coerced."""
    bad = {
        "assignments": [
            {
                "node": "Tanker_Incidents",
                "state": "frequent",
                "reason": "t",
                "state_probs": [{"state": "frequent", "prob": 1.0}],
            }
        ],
        "overall_rationale": "r",
    }
    with pytest.raises(TranslatorError):
        _validate_payload(bad)


def test_validator_rejects_node_outside_snapshot():
    with pytest.raises(TranslatorError):
        _validate_payload(_value_item("Not_A_Real_Node", {"x": 1.0}))


def test_validator_accepts_canonical_value_shape():
    asg, _ = _validate_payload(
        _value_item("Tanker_Incidents", {"none": 0.05, "isolated": 0.3, "frequent": 1.0})
    )
    assert asg[0].node == "Tanker_Incidents"
    assert asg[0].state_probs["frequent"] == 1.0


def test_brace_parser_handles_nested_and_trailing_prose():
    text = (
        'Here is the result:\n```json\n'
        '{"assignments": [{"node": "Tanker_Incidents", "state": "frequent", '
        '"reason": "x {with brace}", "state_probs": [{"state": "frequent", "value": 1.0}]}], '
        '"overall_rationale": "ok"}\n```\nHope that helps!'
    )
    payload = _extract_json_block(text)
    assert payload["overall_rationale"] == "ok"
    assert payload["assignments"][0]["node"] == "Tanker_Incidents"


def test_states_hash_embedded_in_prompt():
    assert _states_hash() in _system_prompt()


# ===== T04 — B1a article-level input + source credibility ==================


def test_translate_article_headline_only_matches_fixture():
    res = translate_article(
        Article(headline="Tanker struck in the Strait of Hormuz"), provider="fake"
    )
    assert any(a.node == "Tanker_Incidents" and a.state == "frequent"
               for a in res.assignments)


def test_translate_headline_is_full_trust_unchanged():
    """A bare headline (analyst paste) is w=1.0 -> ε unchanged from the fixture."""
    res = translate_headline("Tanker struck in the Strait of Hormuz", provider="fake")
    eps = next(a.state_probs for a in res.assignments if a.node == "Tanker_Incidents")
    assert eps == {"none": 0.05, "isolated": 0.30, "frequent": 1.0}


def test_credibility_w0_injects_no_information():
    """w=0 flattens every ε to 1.0 -> a uniform likelihood injects nothing."""
    res = translate_article(
        Article(headline="Tanker struck in the Strait of Hormuz"),
        credibility=0.0, provider="fake",
    )
    eps = next(a.state_probs for a in res.assignments if a.node == "Tanker_Incidents")
    assert all(abs(v - 1.0) < 1e-9 for v in eps.values())


def test_credibility_power_discount_and_maxpin_preserved():
    """ε ← ε**w: best state stays 1.0, others move toward 1.0."""
    res = translate_article(
        Article(headline="Tanker struck in the Strait of Hormuz"),
        credibility=0.5, provider="fake",
    )
    eps = next(a.state_probs for a in res.assignments if a.node == "Tanker_Incidents")
    assert abs(eps["frequent"] - 1.0) < 1e-9               # max-pin preserved
    assert abs(eps["isolated"] - 0.30 ** 0.5) < 1e-9       # discounted
    assert abs(eps["none"] - 0.05 ** 0.5) < 1e-9


def test_source_type_default_credibility_lookup():
    """With no explicit credibility, w is looked up from source_type (state_media=0.3)."""
    assert SOURCE_TYPE_CREDIBILITY["state_media"] == 0.3
    res = translate_article(
        Article(headline="Tanker struck in the Strait of Hormuz",
                source_type="state_media"),
        provider="fake",
    )
    eps = next(a.state_probs for a in res.assignments if a.node == "Tanker_Incidents")
    assert abs(eps["isolated"] - 0.30 ** 0.3) < 1e-9


# ===== T05 — B3 relevance filter + abstention ==============================


def _eps_payload(relevance=None):
    payload = {
        "assignments": [
            {"node": "Tanker_Incidents", "state": "frequent", "reason": "x",
             "state_probs": [{"state": "none", "value": 0.05},
                             {"state": "isolated", "value": 0.3},
                             {"state": "frequent", "value": 1.0}]}
        ],
        "overall_rationale": "r",
    }
    if relevance is not None:
        payload["relevance"] = relevance
    return payload


def test_offtopic_fixture_abstains():
    res = translate_article(Article(headline="Champions League final tonight"), provider="fake")
    assert res.relevance == "no"
    assert res.assignments == []


def test_partial_fixture_kept_and_flagged():
    res = translate_article(
        Article(headline="Brent crude prices climb on Gulf jitters"), provider="fake"
    )
    assert res.relevance == "partial"
    assert any(a.node == "Oil_Price_Regime" for a in res.assignments)


def test_relevance_defaults_yes_when_absent():
    """Pre-B3 recorded payloads (no relevance field) default to 'yes', kept."""
    asg, _rat, rel = _finalize_payload(_eps_payload(relevance=None))
    assert rel == "yes"
    assert asg


def test_relevance_no_drops_assignments():
    """relevance='no' wins: assignments are dropped (abstention)."""
    asg, _rat, rel = _finalize_payload(_eps_payload(relevance="no"))
    assert rel == "no"
    assert asg == []


def test_invalid_relevance_rejected():
    with pytest.raises(TranslatorError):
        _finalize_payload(_eps_payload(relevance="maybe"))
