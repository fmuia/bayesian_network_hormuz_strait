"""Second-order uncertainty for scenario probabilities.

Each elicited CPT column is treated as the mean of a Dirichlet
distribution (multi-state generalisation of Beta) with concentration
parameter `concentration`. We sample M perturbed networks, run inference
on each, and report the central probability plus a credible interval.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from .network import build_network


def _resample_cpd(cpd: TabularCPD, concentration: float, rng: np.random.Generator) -> TabularCPD:
    """Return a new TabularCPD whose columns are Dirichlet-resampled."""
    values = np.asarray(cpd.get_values(), dtype=float)
    n_states, n_cols = values.shape
    new_values = np.empty_like(values)
    for col in range(n_cols):
        alpha = concentration * values[:, col] + 1e-6  # avoid zero-alpha
        new_values[:, col] = rng.dirichlet(alpha)
    return TabularCPD(
        variable=cpd.variable,
        variable_card=n_states,
        values=new_values.tolist(),
        evidence=cpd.variables[1:] if len(cpd.variables) > 1 else None,
        evidence_card=cpd.cardinality[1:].tolist() if len(cpd.cardinality) > 1 else None,
        state_names=cpd.state_names,
    )


def _resampled_network(
    base: DiscreteBayesianNetwork, concentration: float, rng: np.random.Generator
) -> DiscreteBayesianNetwork:
    net = DiscreteBayesianNetwork(list(base.edges()))
    new_cpds = [_resample_cpd(cpd, concentration, rng) for cpd in base.get_cpds()]
    net.add_cpds(*new_cpds)
    return net


def scenario_credible_intervals(
    evidence: Mapping[str, str],
    *,
    m: int = 200,
    concentration: float = 20.0,
    ci: float = 0.80,
    seed: int = 0,
    base_network: DiscreteBayesianNetwork | None = None,
) -> Dict[str, Tuple[float, float, float]]:
    """Mean and (lo, hi) credible interval for each Scenario state.

    Returns ``{scenario: (mean, lo, hi)}``. The mean is the average of
    the M Monte-Carlo posteriors; lo/hi are the empirical quantiles
    bracketing the central `ci` mass.
    """
    if not 0 < ci < 1:
        raise ValueError("ci must be in (0, 1)")
    base = base_network or build_network()
    rng = np.random.default_rng(seed)
    samples: list[Dict[str, float]] = []
    for _ in range(m):
        net = _resampled_network(base, concentration, rng)
        ve = VariableElimination(net)
        f = ve.query(["Scenario"], evidence=dict(evidence), show_progress=False)
        samples.append({s: float(f.values[i]) for i, s in enumerate(f.state_names["Scenario"])})

    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q
    out: Dict[str, Tuple[float, float, float]] = {}
    for scenario in samples[0]:
        arr = np.array([s[scenario] for s in samples])
        out[scenario] = (
            float(arr.mean()),
            float(np.quantile(arr, lo_q)),
            float(np.quantile(arr, hi_q)),
        )
    return out


__all__ = ["scenario_credible_intervals"]
