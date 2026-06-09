"""Contamination probes for AI experts.

Seed calibration is inflated if an agent *recalls* a memorised answer rather than
*reasoning*. These probes test for that rather than assuming its absence
(methodology §8.3). Each probe is a pure function over data the panel collects,
so they are deterministic and testable; the data itself (citations, perturbed
answers, per-model spreads) comes from the agents at run time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class ProbeResult:
    probe: str
    seed_ref: str
    flagged: bool
    detail: dict = field(default_factory=dict)


def source_attribution_probe(seed_ref: str, can_cite_source: bool) -> ProbeResult:
    """Flag if the agent can name where it knows the answer from (i.e. recall)."""
    return ProbeResult("source_attribution", seed_ref, flagged=bool(can_cite_source))


def perturbation_probe(
    seed_ref: str,
    original_answer: ArrayLike,
    perturbed_answer: ArrayLike,
    min_expected_shift: float = 0.1,
) -> ProbeResult:
    """Flag if a perturbation that *should* change the answer barely moves it.

    The shift is the L1 distance between the original and perturbed answers; a
    shift below ``min_expected_shift`` suggests memorised pattern-matching.
    """
    shift = float(np.sum(np.abs(np.asarray(original_answer, float) - np.asarray(perturbed_answer, float))))
    return ProbeResult(
        "perturbation", seed_ref, flagged=shift < min_expected_shift, detail={"l1_shift": shift}
    )


def cross_model_variance_probe(
    seed_ref: str, per_model_values: ArrayLike, min_variance: float = 1e-4
) -> ProbeResult:
    """Flag anomalously low spread across *different* base models on a seed —
    suspicious agreement that suggests a shared memorised answer."""
    var = float(np.var(np.asarray(per_model_values, dtype=float)))
    return ProbeResult(
        "cross_model_variance", seed_ref, flagged=var < min_variance, detail={"variance": var}
    )


def split_calibration_probe(
    in_corpus_calibration: float, post_cutoff_calibration: float, max_gap: float = 0.3
) -> ProbeResult:
    """Flag a large in-corpus minus post-cutoff calibration gap (inflation)."""
    gap = float(in_corpus_calibration - post_cutoff_calibration)
    return ProbeResult(
        "split_calibration", seed_ref="(panel)", flagged=gap > max_gap, detail={"gap": gap}
    )


def summarize_probes(results: list[ProbeResult]) -> dict:
    """Aggregate probe outcomes into a provenance-ready summary."""
    by_probe: dict[str, dict] = {}
    for r in results:
        entry = by_probe.setdefault(r.probe, {"flagged": 0, "total": 0})
        entry["total"] += 1
        entry["flagged"] += int(r.flagged)
    return {
        "by_probe": by_probe,
        "any_flagged": any(r.flagged for r in results),
        "n_flagged": sum(r.flagged for r in results),
    }
