"""The defensible confidence statement.

Confidence in a model output is reported as a *vector, never a scalar*
(methodology §7): the point posterior, a propagated credible interval, the
variance decomposition, the empirical calibration track record (where it
exists), and an IPCC-style confidence rating with an explicit structural-
uncertainty caveat. The interval is always labelled conditional on the model
structure.
"""

from __future__ import annotations

from dataclasses import dataclass

EVIDENCE_LEVELS = {"limited": 0, "medium": 1, "robust": 2}
AGREEMENT_LEVELS = {"low": 0, "medium": 1, "high": 2}
_CONFIDENCE_SCALE = ["very low", "low", "medium", "high", "very high"]


def ipcc_likelihood(probability: float) -> str:
    """Map a probability to IPCC AR5 calibrated likelihood language."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    table = [
        (0.99, "virtually certain"),
        (0.90, "very likely"),
        (0.66, "likely"),
        (0.33, "about as likely as not"),
        (0.10, "unlikely"),
        (0.01, "very unlikely"),
        (0.0, "exceptionally unlikely"),
    ]
    for threshold, phrase in table:
        if probability >= threshold:
            return phrase
    return "exceptionally unlikely"


def ipcc_confidence(evidence: str, agreement: str) -> str:
    """Combine the evidence and agreement dimensions into a confidence rating."""
    if evidence not in EVIDENCE_LEVELS or agreement not in AGREEMENT_LEVELS:
        raise ValueError("unknown evidence/agreement level")
    idx = EVIDENCE_LEVELS[evidence] + AGREEMENT_LEVELS[agreement]
    return _CONFIDENCE_SCALE[idx]


@dataclass(frozen=True)
class ConfidenceReport:
    """All components of the confidence statement. Never reduced to a scalar."""

    query: str
    point: float
    credible_interval: tuple[float, float]
    likelihood: str
    confidence: str
    variance_shares: dict[str, float]
    structural_caveat: str
    empirical: dict | None = None  # Brier / reliability, when outcome data exists

    def summary(self) -> str:
        lo, hi = self.credible_interval
        top = max(self.variance_shares, key=self.variance_shares.get) if self.variance_shares else None
        driver = (
            f" Interval width is dominated by {top} "
            f"({self.variance_shares[top]:.0%})." if top else ""
        )
        empirical = (
            f" Empirical: {self.empirical}." if self.empirical else
            " No out-of-sample validation yet (confidence is model-internal)."
        )
        return (
            f"{self.query}: {self.point:.2f} ({self.likelihood}), "
            f"90% credible interval [{lo:.2f}, {hi:.2f}], {self.confidence} confidence. "
            f"Conditional on model structure: {self.structural_caveat}.{driver}{empirical}"
        )


def assemble_confidence(
    query: str,
    point: float,
    credible_interval: tuple[float, float],
    variance_shares: dict[str, float],
    evidence: str,
    agreement: str,
    structural_caveat: str = "the latent-regime DAG is assumed correct",
    empirical: dict | None = None,
) -> ConfidenceReport:
    """Assemble the four-component confidence statement."""
    return ConfidenceReport(
        query=query,
        point=point,
        credible_interval=credible_interval,
        likelihood=ipcc_likelihood(point),
        confidence=ipcc_confidence(evidence, agreement),
        variance_shares=variance_shares,
        structural_caveat=structural_caveat,
        empirical=empirical,
    )


__all__ = [
    "ConfidenceReport",
    "assemble_confidence",
    "ipcc_likelihood",
    "ipcc_confidence",
]
