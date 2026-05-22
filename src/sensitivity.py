"""Second-order uncertainty for scenario probabilities.

Each elicited CPT column is treated as the mean of a Dirichlet
distribution (multi-state generalisation of Beta) with concentration
parameter `concentration`. We sample M perturbed networks, run inference
on each, and report the central probability plus a credible interval.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from .network import STATES, build_network


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


def _virtual_evidence_cpds(
    soft_evidence: Mapping[str, Mapping[str, float]],
) -> list[TabularCPD]:
    cpds: list[TabularCPD] = []
    for node, dist in soft_evidence.items():
        vals = [[float(dist.get(s, 0.0))] for s in STATES[node]]
        cpds.append(
            TabularCPD(
                variable=node,
                variable_card=len(STATES[node]),
                values=vals,
                state_names={node: STATES[node]},
            )
        )
    return cpds


def node_credible_intervals(
    evidence: Mapping[str, str],
    *,
    nodes: Optional[Sequence[str]] = None,
    soft_evidence: Optional[Mapping[str, Mapping[str, float]]] = None,
    m: int = 200,
    concentration: float = 20.0,
    ci: float = 0.80,
    seed: int = 0,
    base_network: Optional[DiscreteBayesianNetwork] = None,
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """Per-state credible intervals for one or more nodes.

    Returns ``{node: {state: (mean, lo, hi)}}``. Hard-observed nodes are
    returned as deltas (``(1, 1, 1)`` on the observed state, ``(0, 0, 0)``
    elsewhere) without sampling; all other nodes share a single resample
    loop so cost is O(m) network constructions regardless of how many
    nodes are requested.
    """
    if not 0 < ci < 1:
        raise ValueError("ci must be in (0, 1)")

    target_nodes = list(nodes) if nodes is not None else list(STATES.keys())
    for node in target_nodes:
        if node not in STATES:
            raise KeyError(f"Unknown node: {node}")

    hard = dict(evidence)
    soft = {k: dict(v) for k, v in (soft_evidence or {}).items()}

    hard_targets = [n for n in target_nodes if n in hard]
    sampled_targets = [n for n in target_nodes if n not in hard]

    out: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    for node in hard_targets:
        observed = hard[node]
        out[node] = {
            s: (1.0, 1.0, 1.0) if s == observed else (0.0, 0.0, 0.0)
            for s in STATES[node]
        }

    if not sampled_targets:
        return out

    base = base_network or build_network()
    rng = np.random.default_rng(seed)
    virtual_ev = _virtual_evidence_cpds(soft)

    samples: Dict[str, list[Dict[str, float]]] = {n: [] for n in sampled_targets}
    for _ in range(m):
        net = _resampled_network(base, concentration, rng)
        ve = VariableElimination(net)
        for node in sampled_targets:
            kwargs: Dict[str, object] = {
                "evidence": dict(hard),
                "show_progress": False,
            }
            if virtual_ev:
                kwargs["virtual_evidence"] = virtual_ev
            factor = ve.query([node], **kwargs)
            states = factor.state_names[node]
            samples[node].append(
                {s: float(factor.values[i]) for i, s in enumerate(states)}
            )

    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q
    for node in sampled_targets:
        node_samples = samples[node]
        out[node] = {}
        for state in STATES[node]:
            arr = np.array([sample[state] for sample in node_samples])
            out[node][state] = (
                float(arr.mean()),
                float(np.quantile(arr, lo_q)),
                float(np.quantile(arr, hi_q)),
            )
    return out


__all__ = ["scenario_credible_intervals", "node_credible_intervals"]
