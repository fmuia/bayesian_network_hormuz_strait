"""Core engine: aggregation, calibration scoring, kappa mapping, sensitivity.

Pure functions with no database or UI dependencies (Plan 4, Layer 1).
"""

from __future__ import annotations

from .aggregation import cooke_pool, linear_pool, logarithmic_pool
from .calibration import (
    ExpertScore,
    bin_probabilities,
    calibration_score,
    classical_model_weights,
    information_score,
    intrinsic_range,
)
from .kappa import (
    KappaLadder,
    cap_level,
    dirichlet_variance,
    kappa_from_panel_spread,
    kappa_from_seed_coverage,
    snap_to_level,
)
from .sensitivity import (
    MorrisResult,
    SobolResult,
    VarianceDecomposition,
    morris_screening,
    posterior_variance_decomposition,
    sobol_indices,
)

__all__ = [
    "linear_pool",
    "logarithmic_pool",
    "cooke_pool",
    "calibration_score",
    "information_score",
    "classical_model_weights",
    "intrinsic_range",
    "bin_probabilities",
    "ExpertScore",
    "kappa_from_panel_spread",
    "kappa_from_seed_coverage",
    "snap_to_level",
    "cap_level",
    "dirichlet_variance",
    "KappaLadder",
    "morris_screening",
    "sobol_indices",
    "posterior_variance_decomposition",
    "MorrisResult",
    "SobolResult",
    "VarianceDecomposition",
]
