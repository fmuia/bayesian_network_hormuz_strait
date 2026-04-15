"""Inference wrapper for the Strait of Hormuz Bayesian network."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from .network import STATES, build_network


class BNInferenceEngine:
    """Stateful wrapper that accumulates evidence and answers queries.

    The engine owns a network, a `VariableElimination` instance, and an
    evidence dict. Mutating evidence does not require rebuilding the
    network; pgmpy's VE handles per-query evidence pass-through.
    """

    def __init__(self, network: Optional[DiscreteBayesianNetwork] = None) -> None:
        self._network: DiscreteBayesianNetwork = network or build_network()
        self._engine = VariableElimination(self._network)
        self._evidence: Dict[str, str] = {}
        self._soft_evidence: Dict[str, Dict[str, float]] = {}

    # -- evidence management -------------------------------------------------

    def update_evidence(self, evidence: Dict[str, str]) -> None:
        """Merge `evidence` into the current evidence set.

        Validates node and state names against the network's vocabulary.
        """
        for node, state in evidence.items():
            if node not in STATES:
                raise KeyError(f"Unknown node: {node}")
            if state not in STATES[node]:
                raise ValueError(
                    f"Invalid state {state!r} for node {node!r}; "
                    f"valid: {STATES[node]}"
                )
            self._soft_evidence.pop(node, None)
            self._evidence[node] = state

    def update_soft_evidence(
        self, soft_evidence: Mapping[str, Mapping[str, float]]
    ) -> None:
        for node, dist in soft_evidence.items():
            if node not in STATES:
                raise KeyError(f"Unknown node: {node}")
            probs: Dict[str, float] = {}
            for state in STATES[node]:
                p = float(dist.get(state, 0.0))
                if p < 0.0:
                    raise ValueError(f"Negative probability for {node}.{state}: {p}")
                probs[state] = p
            total = sum(probs.values())
            if total <= 0.0:
                raise ValueError(f"Soft evidence for {node} sums to zero.")
            probs = {k: v / total for k, v in probs.items()}
            self._evidence.pop(node, None)
            self._soft_evidence[node] = probs

    def clear_evidence(self) -> None:
        """Reset to prior (no evidence)."""
        self._evidence.clear()
        self._soft_evidence.clear()

    @property
    def evidence(self) -> Dict[str, str]:
        """Read-only view of current evidence."""
        return dict(self._evidence)

    @property
    def network(self) -> DiscreteBayesianNetwork:
        return self._network

    # -- queries -------------------------------------------------------------

    def get_prior_probabilities(self) -> Dict[str, float]:
        """Scenario marginal with no evidence applied."""
        result = self._engine.query(["Scenario"], evidence={}, show_progress=False)
        return self._distribution(result, "Scenario")

    def get_scenario_probabilities(self) -> Dict[str, float]:
        """Scenario marginal under the current accumulated evidence."""
        result = self._engine.query(
            ["Scenario"],
            evidence=self._evidence,
            virtual_evidence=self._virtual_evidence_cpds(),
            show_progress=False,
        )
        return self._distribution(result, "Scenario")

    def get_node_marginal(self, node: str) -> Dict[str, float]:
        """Marginal distribution of any node under current evidence.

        Returns the directly-observed delta if the node is in evidence.
        """
        if node not in STATES:
            raise KeyError(f"Unknown node: {node}")
        if node in self._evidence:
            return {s: (1.0 if s == self._evidence[node] else 0.0) for s in STATES[node]}
        ev = {k: v for k, v in self._evidence.items() if k != node}
        ve = self._virtual_evidence_cpds()
        if ve:
            result = self._engine.query(
                [node], evidence=ev, virtual_evidence=ve, show_progress=False
            )
        else:
            result = self._engine.query([node], evidence=ev, show_progress=False)
        return self._distribution(result, node)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _distribution(factor, node: str) -> Dict[str, float]:
        states = factor.state_names[node]
        values = factor.values
        return {state: float(values[i]) for i, state in enumerate(states)}

    def _virtual_evidence_cpds(
        self, *, exclude_node: Optional[str] = None
    ) -> list[TabularCPD]:
        cpds: list[TabularCPD] = []
        for node, dist in self._soft_evidence.items():
            if exclude_node is not None and node == exclude_node:
                continue
            vals = [[dist[s]] for s in STATES[node]]
            cpds.append(
                TabularCPD(
                    variable=node,
                    variable_card=len(STATES[node]),
                    values=vals,
                    state_names={node: STATES[node]},
                )
            )
        return cpds


__all__ = ["BNInferenceEngine"]
