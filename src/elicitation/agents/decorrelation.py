"""Inter-agent correlation and effective sample size.

Agents off the same base model are not independent, so their apparent diversity
overstates the information in the panel. We measure mean pairwise correlation
across agents' answer series, convert to an effective sample size, and feed that
into the kappa estimate (which widens accordingly). Roles/personas are *not*
credited as independence — only distinct base models are (methodology §8.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


def mean_pairwise_correlation(answer_series: ArrayLike) -> float:
    """Mean off-diagonal Pearson correlation across agents' answer series.

    ``answer_series`` is (n_agents, n_items). Constant series (undefined
    correlation) contribute 0. Negative correlations are kept in the mean but the
    consumer clips to [0, 1] for the discount.
    """
    a = np.asarray(answer_series, dtype=float)
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 2:
        return 0.0
    corr = np.corrcoef(a)
    corr = np.nan_to_num(corr, nan=0.0)
    n = corr.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.mean(corr[iu]))


def effective_sample_size(n: int, correlation: float) -> float:
    """n / (1 + (n-1) rho), with rho clipped to [0, 1]."""
    rho = float(np.clip(correlation, 0.0, 1.0))
    return n / (1.0 + (n - 1) * rho)


@dataclass(frozen=True)
class CorrelationAdjustment:
    mean_correlation: float
    n_agents: int
    effective_n: float
    distinct_base_models: int

    @property
    def note(self) -> str:
        return (
            f"mean pairwise correlation {self.mean_correlation:.2f}; "
            f"{self.n_agents} agents across {self.distinct_base_models} base models; "
            f"effective N {self.effective_n:.2f}"
        )
