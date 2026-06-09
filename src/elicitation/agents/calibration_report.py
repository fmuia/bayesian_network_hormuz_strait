"""Assemble the panel-level calibration report written to ``cpt_provenance``.

Every AI-sourced CPT carries calibration scores, the κ level, the model set, an
inter-agent correlation note, and a contamination-probe summary — the hard
defensibility requirement (Plan 4, Layer 3 validation).
"""

from __future__ import annotations

from typing import Sequence

from ..protocols.base import Expert
from ..protocols.cooke import CookeResult
from .decorrelation import CorrelationAdjustment


def panel_calibration_report(
    experts: Sequence[Expert],
    cooke_result: CookeResult,
    correlation: CorrelationAdjustment,
    contamination_summary: dict | None = None,
) -> dict:
    """Build the provenance-ready report for a panel run."""
    per_expert = [
        {
            "name": e.name,
            "kind": e.kind,
            "identity": {"base_model": e.base_model, "role": e.role, "config": e.config_fingerprint},
            "calibration": score.calibration,
            "information": score.information,
            "weight": score.weight,
        }
        for e, score in zip(experts, cooke_result.expert_scores)
    ]
    return {
        "protocol": "cooke",
        "is_ai_sourced": any(e.kind == "ai" for e in experts),
        "model_set": {
            "models": sorted({e.base_model for e in experts if e.base_model}),
            "n_experts": len(experts),
            "roles": sorted({e.role for e in experts if e.role}),
        },
        "experts": per_expert,
        "correlation": {
            "mean_correlation": correlation.mean_correlation,
            "effective_n": correlation.effective_n,
            "distinct_base_models": correlation.distinct_base_models,
            "note": correlation.note,
        },
        "contamination": contamination_summary or {},
    }
