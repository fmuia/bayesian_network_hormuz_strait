"""Inference wrapper for the Strait of Hormuz Bayesian network."""

from __future__ import annotations

from typing import Dict, Optional

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
            self._evidence[node] = state

    def clear_evidence(self) -> None:
        """Reset to prior (no evidence)."""
        self._evidence.clear()

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
            ["Scenario"], evidence=self._evidence, show_progress=False
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
        result = self._engine.query([node], evidence=ev, show_progress=False)
        return self._distribution(result, node)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _distribution(factor, node: str) -> Dict[str, float]:
        states = factor.state_names[node]
        values = factor.values
        return {state: float(values[i]) for i, state in enumerate(states)}


__all__ = ["BNInferenceEngine"]
