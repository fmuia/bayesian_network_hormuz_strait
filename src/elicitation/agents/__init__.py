"""AI experts and multi-model panels (Plan 4, Layers 2-3)."""

from __future__ import annotations

from .calibration_report import panel_calibration_report
from .contamination import (
    ProbeResult,
    cross_model_variance_probe,
    perturbation_probe,
    source_attribution_probe,
    split_calibration_probe,
    summarize_probes,
)
from .decorrelation import (
    CorrelationAdjustment,
    effective_sample_size,
    mean_pairwise_correlation,
)
from .llm_expert import (
    CompletionClient,
    LLMExpert,
    OpenAICompletionClient,
    ScriptedCompletionClient,
    normalize,
)
from .panel import Panel, PanelResult, ensure_consistent_identity

__all__ = [
    # experts
    "LLMExpert",
    "CompletionClient",
    "ScriptedCompletionClient",
    "OpenAICompletionClient",
    "normalize",
    # decorrelation
    "mean_pairwise_correlation",
    "effective_sample_size",
    "CorrelationAdjustment",
    # contamination
    "ProbeResult",
    "source_attribution_probe",
    "perturbation_probe",
    "cross_model_variance_probe",
    "split_calibration_probe",
    "summarize_probes",
    # panel
    "Panel",
    "PanelResult",
    "ensure_consistent_identity",
    "panel_calibration_report",
]
