"""Tests for the Cooke protocol and the LLM expert (Layer 2)."""

from __future__ import annotations

import numpy as np
import pytest

from src.elicitation.agents import (
    LLMExpert,
    OpenAICompletionClient,
    ScriptedCompletionClient,
    normalize,
)
from src.elicitation.protocols import (
    CookeProtocol,
    CPTColumnTarget,
    ScriptedExpert,
    SeedQuestion,
)

# --------------------------------------------------------------------------- #
# Target validation
# --------------------------------------------------------------------------- #


def test_cpt_column_target_validation() -> None:
    t = CPTColumnTarget(node="T", states=("none", "isolated", "frequent"), parent_config=("crisis",))
    assert t.n_outcomes() == 3
    assert "crisis" in t.describe()
    t.validate_distribution([0.1, 0.3, 0.6])
    with pytest.raises(ValueError):
        t.validate_distribution([0.1, 0.3])
    with pytest.raises(ValueError):
        t.validate_distribution([0.1, 0.3, 0.3])
    with pytest.raises(ValueError):
        CPTColumnTarget(node="x", states=("only",))


# --------------------------------------------------------------------------- #
# Cooke end-to-end with scripted (human-like) experts
# --------------------------------------------------------------------------- #


def _seeds(rng: np.random.Generator, n: int = 200) -> list[SeedQuestion]:
    realizations = rng.normal(0.0, 1.0, size=n)
    return [SeedQuestion(id=f"s{i}", text=f"seed {i}", realization=float(r)) for i, r in enumerate(realizations)]


def _calibrated_seed_answers(seeds, q=(-1.645, 0.0, 1.645)) -> dict:
    return {s.id: q for s in seeds}


def _overconfident_seed_answers(seeds, q=(2.0, 3.0, 4.0)) -> dict:
    return {s.id: q for s in seeds}


def test_cooke_run_zeroes_bad_expert_and_pools_toward_good() -> None:
    rng = np.random.default_rng(0)
    seeds = _seeds(rng)
    target = CPTColumnTarget(node="T", states=("none", "isolated", "frequent"), parent_config=("crisis",))

    good1 = ScriptedExpert("good1", _calibrated_seed_answers(seeds), {"T": (0.10, 0.30, 0.60)})
    good2 = ScriptedExpert("good2", _calibrated_seed_answers(seeds), {"T": (0.12, 0.34, 0.54)})
    bad = ScriptedExpert("bad", _overconfident_seed_answers(seeds), {"T": (0.60, 0.30, 0.10)})

    result = CookeProtocol().run([good1, good2, bad], seeds, [target], alpha=0.05)

    weights = {e.name: w for e, w in zip([good1, good2, bad], result.weights)}
    assert weights["bad"] == 0.0
    assert weights["good1"] > 0 and weights["good2"] > 0

    mean = np.array(result.targets["T"].mean)
    # pooled toward the good experts (frequent dominant), away from the bad one
    assert mean[2] > mean[0]
    assert mean[2] > 0.5


def test_cooke_assigns_kappa_level_and_provenance() -> None:
    rng = np.random.default_rng(1)
    seeds = _seeds(rng)
    target = CPTColumnTarget(node="T", states=("a", "b", "c"))
    # two well-calibrated, closely-agreeing experts -> tighter kappa
    e1 = ScriptedExpert("e1", _calibrated_seed_answers(seeds), {"T": (0.20, 0.30, 0.50)})
    e2 = ScriptedExpert("e2", _calibrated_seed_answers(seeds), {"T": (0.21, 0.31, 0.48)})
    res = CookeProtocol().run([e1, e2], seeds, [target], alpha=0.0)
    tr = res.targets["T"]
    assert tr.kappa_level in {"tight", "normal", "uncertain"}
    assert tr.provenance.protocol == "cooke"
    assert tr.provenance.is_ai_sourced is False
    assert set(tr.provenance.weights) == {"e1", "e2"}


def test_kappa_capped_when_panel_poorly_calibrated() -> None:
    """A panel that fails the seeds cannot claim a tight kappa, however much it
    agrees on the target."""
    rng = np.random.default_rng(2)
    seeds = _seeds(rng)
    target = CPTColumnTarget(node="T", states=("a", "b", "c"))
    # both overconfident on seeds (low calibration), but agree tightly on target
    bad1 = ScriptedExpert("b1", _overconfident_seed_answers(seeds), {"T": (0.33, 0.33, 0.34)})
    bad2 = ScriptedExpert("b2", _overconfident_seed_answers(seeds), {"T": (0.33, 0.34, 0.33)})
    tr = CookeProtocol().run([bad1, bad2], seeds, [target], alpha=0.0).targets["T"]
    assert tr.kappa_level == "uncertain"  # capped by poor calibration


# --------------------------------------------------------------------------- #
# LLM expert drives Cooke (integration via the deterministic fake client)
# --------------------------------------------------------------------------- #


def test_llm_expert_drives_cooke_and_is_flagged_ai_sourced() -> None:
    rng = np.random.default_rng(3)
    seeds = _seeds(rng, n=120)
    target = CPTColumnTarget(node="T", states=("a", "b", "c"))

    client = ScriptedCompletionClient(
        seed_answers={s.text: (-1.645, 0.0, 1.645) for s in seeds},
        target_answers={"T": (0.2, 0.3, 0.5)},
    )
    a = LLMExpert("claude-agent", base_model="claude", client=client, role="base-rate-thinker")
    b = LLMExpert("gpt-agent", base_model="gpt", client=client, role="red-team")

    res = CookeProtocol().run([a, b], seeds, [target], alpha=0.0)
    tr = res.targets["T"]
    assert tr.provenance.is_ai_sourced is True
    assert set(tr.provenance.model_set["models"]) == {"claude", "gpt"}
    assert "red-team" in tr.provenance.model_set["roles"]
    np.testing.assert_allclose(tr.mean, [0.2, 0.3, 0.5], atol=1e-9)


def test_llm_expert_identity_includes_role_and_config() -> None:
    client = ScriptedCompletionClient({}, {})
    e1 = LLMExpert("x", base_model="claude", client=client, role="hawk", config={"temp": 0.0})
    e2 = LLMExpert("y", base_model="claude", client=client, role="dove", config={"temp": 0.0})
    # same model, different role -> different identity tuple
    assert e1.identity != e2.identity
    assert e1.identity[0] == "claude" and e1.identity[1] == "hawk"


# --------------------------------------------------------------------------- #
# Adapter helpers (no network)
# --------------------------------------------------------------------------- #


def test_normalize_handles_unnormalised_and_negative() -> None:
    np.testing.assert_allclose(normalize([2, 2, 4]), [0.25, 0.25, 0.5])
    np.testing.assert_allclose(normalize([-1, 1, 1]), [0.0, 0.5, 0.5])
    with pytest.raises(ValueError):
        normalize([0, 0, 0])


def test_openai_adapter_prompt_and_parser() -> None:
    prompt = OpenAICompletionClient.build_distribution_prompt(
        "P(T | crisis)", ["none", "isolated", "frequent"], role="red-team"
    )
    assert "red-team" in prompt and "3 probabilities" in prompt
    assert OpenAICompletionClient.parse_float_array('here you go: [0.1, 0.3, 0.6] done') == [0.1, 0.3, 0.6]
    with pytest.raises(ValueError):
        OpenAICompletionClient.parse_float_array("no array here")
