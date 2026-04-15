"""Sanity tests for the BN definition, inference, and translator schema."""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock

import pytest

from src.evidence import EXAMPLE_HEADLINES
from src.inference import BNInferenceEngine
from src.network import EDGES, STATES, build_network
from src.translator import (
    TranslatorError,
    _node_state_enum_schema,
    _system_prompt,
    translate_headline,
)


# -- Network structure -------------------------------------------------------


def test_network_builds_and_validates() -> None:
    net = build_network()
    assert net.check_model() is True
    assert set(net.nodes()) == set(STATES.keys())
    assert set(net.edges()) == set(EDGES)


def test_all_cpts_normalised() -> None:
    net = build_network()
    for cpd in net.get_cpds():
        values = cpd.get_values()
        for col in range(values.shape[1]):
            assert math.isclose(values[:, col].sum(), 1.0, abs_tol=1e-8)


# -- Inference ---------------------------------------------------------------


def test_prior_scenario_distribution() -> None:
    engine = BNInferenceEngine()
    prior = engine.get_prior_probabilities()
    assert set(prior.keys()) == set(STATES["Scenario"])
    assert math.isclose(sum(prior.values()), 1.0, abs_tol=1e-8)
    for p in prior.values():
        assert 0.0 < p < 1.0


def test_extreme_escalation_raises_severe_closure() -> None:
    engine = BNInferenceEngine()
    prior_severe = engine.get_prior_probabilities()["Severe_Closure"]
    engine.update_evidence({
        "US_Iran_Negotiations": "breakdown",
        "Sanctions_Trajectory": "tightening",
        "Iranian_Regime_Stability": "unstable",
        "Tanker_Incidents": "frequent",
        "US_Military_Response": "major",
    })
    posterior_severe = engine.get_scenario_probabilities()["Severe_Closure"]
    assert posterior_severe > prior_severe + 0.10


def test_extreme_deescalation_raises_stress_mitigates() -> None:
    engine = BNInferenceEngine()
    prior_mit = engine.get_prior_probabilities()["Stress_Mitigates"]
    engine.update_evidence({
        "US_Iran_Negotiations": "success",
        "Third_Party_Mediation": "active",
        "Iranian_Regime_Stability": "stable",
        "Tanker_Incidents": "none",
        "US_Military_Response": "none",
    })
    posterior_mit = engine.get_scenario_probabilities()["Stress_Mitigates"]
    assert posterior_mit > prior_mit + 0.05


def test_clear_evidence_returns_to_prior() -> None:
    engine = BNInferenceEngine()
    prior = engine.get_prior_probabilities()
    engine.update_evidence({"US_Iran_Negotiations": "breakdown"})
    engine.clear_evidence()
    after = engine.get_scenario_probabilities()
    for k in prior:
        assert math.isclose(prior[k], after[k], abs_tol=1e-8)


def test_node_marginal_observed_node_is_delta() -> None:
    engine = BNInferenceEngine()
    engine.update_evidence({"Strait_Operationally_Closed": "full"})
    m = engine.get_node_marginal("Strait_Operationally_Closed")
    assert math.isclose(m["full"], 1.0, abs_tol=1e-8)
    assert math.isclose(m["no"] + m["partial"], 0.0, abs_tol=1e-8)


# -- Example headlines -------------------------------------------------------


def test_example_headlines_well_formed() -> None:
    assert len(EXAMPLE_HEADLINES) >= 3
    for ex in EXAMPLE_HEADLINES:
        assert ex.text and isinstance(ex.text, str)
        assert ex.tone in {"de-escalation", "mixed", "escalation"}


# -- Translator --------------------------------------------------------------


def test_translator_schema_uses_valid_nodes() -> None:
    schema = _node_state_enum_schema()
    node_enum = schema["properties"]["assignments"]["items"]["properties"]["node"]["enum"]
    assert set(node_enum) == set(STATES.keys())
    assert schema["additionalProperties"] is False


def test_translator_system_prompt_mentions_every_observable_node() -> None:
    prompt = _system_prompt()
    for node in STATES:
        if node == "Scenario":
            continue
        assert node in prompt


def _mock_client(payload: dict) -> MagicMock:
    """Build a MagicMock matching the openai chat.completions.create shape."""
    client = MagicMock()
    message = MagicMock()
    message.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    return client


def test_translator_parses_valid_response() -> None:
    payload = {
        "assignments": [
            {
                "node": "US_Military_Response",
                "state": "major",
                "reason": "Strikes imply a major response.",
            },
            {
                "node": "Diplomatic_Resolution_Path",
                "state": "narrowing",
                "reason": "Strikes narrow the diplomatic window.",
            },
        ],
        "overall_rationale": "US kinetic action tightens the situation.",
    }
    result = translate_headline(
        "US strikes IRGC naval assets", client=_mock_client(payload)
    )
    assert len(result.assignments) == 2
    ev = result.as_evidence_dict()
    assert ev["US_Military_Response"] == "major"
    assert ev["Diplomatic_Resolution_Path"] == "narrowing"


def test_translator_rejects_invalid_state() -> None:
    payload = {
        "assignments": [
            {"node": "US_Military_Response", "state": "catastrophic", "reason": "x"},
        ],
        "overall_rationale": "bad state",
    }
    with pytest.raises(TranslatorError):
        translate_headline("anything", client=_mock_client(payload))


def test_translator_rejects_unknown_node() -> None:
    payload = {
        "assignments": [
            {"node": "Made_Up_Node", "state": "foo", "reason": "x"},
        ],
        "overall_rationale": "bad node",
    }
    with pytest.raises(TranslatorError):
        translate_headline("anything", client=_mock_client(payload))


def test_translator_skips_scenario_node_silently() -> None:
    payload = {
        "assignments": [
            {"node": "Scenario", "state": "Severe_Closure", "reason": "leak"},
            {
                "node": "Tanker_Incidents",
                "state": "frequent",
                "reason": "Multiple attacks.",
            },
        ],
        "overall_rationale": "Scenario leaked; dropped.",
    }
    result = translate_headline("anything", client=_mock_client(payload))
    assert [a.node for a in result.assignments] == ["Tanker_Incidents"]


def test_translator_empty_headline_errors() -> None:
    with pytest.raises(TranslatorError):
        translate_headline("   ", client=_mock_client({"assignments": [], "overall_rationale": ""}))
