"""Tests for Layer 3: decorrelation, contamination probes, panels."""

from __future__ import annotations

import numpy as np
import pytest

from src.elicitation.agents import (
    LLMExpert,
    Panel,
    ScriptedCompletionClient,
    cross_model_variance_probe,
    effective_sample_size,
    ensure_consistent_identity,
    mean_pairwise_correlation,
    perturbation_probe,
    source_attribution_probe,
    split_calibration_probe,
    summarize_probes,
)
from src.elicitation.protocols import CPTColumnTarget, SeedQuestion

# --------------------------------------------------------------------------- #
# Decorrelation
# --------------------------------------------------------------------------- #


def test_correlation_high_for_identical_low_for_independent() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=30)
    identical = np.array([base, base + 1e-9])  # same shape, varying across items
    assert mean_pairwise_correlation(identical) > 0.99
    independent = rng.normal(size=(2, 30))
    assert abs(mean_pairwise_correlation(independent)) < 0.5


def test_effective_sample_size_drops_with_correlation() -> None:
    assert effective_sample_size(2, 0.0) == pytest.approx(2.0)
    assert effective_sample_size(2, 1.0) == pytest.approx(1.0)
    assert effective_sample_size(4, 0.5) < 4.0
    # monotonic decreasing in rho
    vals = [effective_sample_size(5, r) for r in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


# --------------------------------------------------------------------------- #
# Contamination probes
# --------------------------------------------------------------------------- #


def test_source_attribution_probe() -> None:
    assert source_attribution_probe("s1", can_cite_source=True).flagged is True
    assert source_attribution_probe("s1", can_cite_source=False).flagged is False


def test_perturbation_probe_flags_unchanged_answer() -> None:
    # answer barely moves when the perturbation should have changed it -> recall
    flagged = perturbation_probe("s1", [0.2, 0.3, 0.5], [0.2, 0.3, 0.5])
    assert flagged.flagged is True
    moved = perturbation_probe("s1", [0.2, 0.3, 0.5], [0.6, 0.2, 0.2])
    assert moved.flagged is False


def test_cross_model_variance_probe_flags_suspicious_agreement() -> None:
    assert cross_model_variance_probe("s1", [0.30, 0.30, 0.30]).flagged is True
    assert cross_model_variance_probe("s1", [0.10, 0.40, 0.70]).flagged is False


def test_split_calibration_probe_flags_inflation() -> None:
    assert split_calibration_probe(in_corpus_calibration=0.9, post_cutoff_calibration=0.4).flagged is True
    assert split_calibration_probe(0.6, 0.55).flagged is False


def test_summarize_probes() -> None:
    results = [
        source_attribution_probe("s1", True),
        source_attribution_probe("s2", False),
        cross_model_variance_probe("s1", [0.3, 0.3, 0.3]),
    ]
    summary = summarize_probes(results)
    assert summary["any_flagged"] is True
    assert summary["n_flagged"] == 2
    assert summary["by_probe"]["source_attribution"] == {"flagged": 1, "total": 2}


# --------------------------------------------------------------------------- #
# Panel orchestration
# --------------------------------------------------------------------------- #


def _seeds(n: int = 60) -> list[SeedQuestion]:
    rng = np.random.default_rng(1)
    return [SeedQuestion(id=f"s{i}", text=f"seed {i}", realization=float(rng.normal())) for i in range(n)]


def _client(seed_q_by_text, target) -> ScriptedCompletionClient:
    return ScriptedCompletionClient(seed_answers=seed_q_by_text, target_answers={"T": target})


def test_panel_detects_multi_model_and_red_team() -> None:
    seeds = _seeds()
    # each agent's per-seed q50 varies across seeds (so correlation is defined)
    qmap = {s.text: (s.realization - 1.6, s.realization, s.realization + 1.6) for s in seeds}
    client = _client(qmap, (0.2, 0.3, 0.5))
    a = LLMExpert("a", base_model="claude", client=client, role="base-rate-thinker")
    b = LLMExpert("b", base_model="gpt", client=client, role="red-team")
    panel = Panel([a, b])
    assert panel.is_multi_model() is True
    assert panel.has_red_team() is True


def test_panel_run_produces_report_with_defensibility_fields() -> None:
    seeds = _seeds()
    qmap_a = {s.text: (s.realization - 1.6, s.realization, s.realization + 1.6) for s in seeds}
    qmap_b = {s.text: (s.realization - 1.7, s.realization + 0.05, s.realization + 1.5) for s in seeds}
    ca = ScriptedCompletionClient(qmap_a, {"T": (0.2, 0.3, 0.5)})
    cb = ScriptedCompletionClient(qmap_b, {"T": (0.22, 0.31, 0.47)})
    a = LLMExpert("a", base_model="claude", client=ca, role="base-rate-thinker")
    b = LLMExpert("b", base_model="gpt", client=cb, role="red-team")
    target = CPTColumnTarget(node="T", states=("none", "isolated", "frequent"), parent_config=("crisis",))

    result = Panel([a, b]).run_cooke(seeds, [target], alpha=0.0)
    report = result.report
    assert report["is_ai_sourced"] is True
    assert set(report["model_set"]["models"]) == {"claude", "gpt"}
    assert "red-team" in report["model_set"]["roles"]
    assert "mean_correlation" in report["correlation"]
    assert "effective_n" in report["correlation"]
    assert len(report["experts"]) == 2
    assert result.cooke.targets["T"].kappa_level in {"tight", "normal", "uncertain"}


def test_identity_guard_rejects_role_mismatch() -> None:
    client = ScriptedCompletionClient({}, {})
    neutral = LLMExpert("x", base_model="claude", client=client, role=None, config={"t": 0})
    hawk = LLMExpert("x", base_model="claude", client=client, role="hawk", config={"t": 0})
    ensure_consistent_identity(neutral, neutral)  # ok
    with pytest.raises(ValueError):
        ensure_consistent_identity(neutral, hawk)  # neutral score can't weight an in-role estimate
