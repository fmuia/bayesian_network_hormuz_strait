"""Declarative network specification (discrete subset).

A backend-agnostic description of a discrete Bayesian network: nodes, their
states and parents, the CPT values, and — the Plan 4 addition — a per-CPT
``kappa`` (calibration-derived concentration) carried through to the inference
layer. This is the discrete subset authored here (Plan 4, Layer 5); the PyMC
integration extends the same file with continuous nodes and a backend
abstraction.

``to_pgmpy`` / ``from_pgmpy`` bridge to the existing pgmpy engine so elicited
CPTs round-trip with the running model, and ``kappa_map`` feeds
``src.sensitivity`` the per-CPT concentrations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Mapping, Sequence

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

# A CPT is keyed by the parent-state tuple; root nodes use the empty tuple.
CptTable = dict[tuple[str, ...], list[float]]


@dataclass
class DiscreteNode:
    """One discrete node: states, parents, CPT, and its per-CPT kappa."""

    name: str
    states: tuple[str, ...]
    parents: tuple[str, ...] = ()
    cpt: CptTable = field(default_factory=dict)
    kappa: float | None = None
    kappa_level: str | None = None

    def validate(self, all_states: Mapping[str, Sequence[str]]) -> None:
        parent_states = [list(all_states[p]) for p in self.parents]
        expected_keys = (
            [tuple(c) for c in product(*parent_states)] if self.parents else [()]
        )
        if set(self.cpt) != set(expected_keys):
            raise ValueError(
                f"node {self.name}: CPT keys do not match parent configurations"
            )
        for key, col in self.cpt.items():
            if len(col) != len(self.states):
                raise ValueError(f"node {self.name} column {key}: wrong length")
            if abs(sum(col) - 1.0) > 1e-6:
                raise ValueError(f"node {self.name} column {key}: does not sum to 1")


@dataclass
class NetworkSpec:
    """A discrete network: an ordered set of :class:`DiscreteNode`."""

    nodes: dict[str, DiscreteNode] = field(default_factory=dict)

    def add(self, node: DiscreteNode) -> "NetworkSpec":
        self.nodes[node.name] = node
        return self

    @property
    def states(self) -> dict[str, tuple[str, ...]]:
        return {n.name: n.states for n in self.nodes.values()}

    def edges(self) -> list[tuple[str, str]]:
        return [(p, n.name) for n in self.nodes.values() for p in n.parents]

    def kappa_map(self) -> dict[str, float]:
        """Per-node kappa for the nodes that carry one (Plan 4, Layer 5)."""
        return {n.name: float(n.kappa) for n in self.nodes.values() if n.kappa is not None}

    def validate(self) -> None:
        all_states = self.states
        for node in self.nodes.values():
            for p in node.parents:
                if p not in self.nodes:
                    raise ValueError(f"node {node.name}: unknown parent {p}")
            node.validate(all_states)

    # -- pgmpy bridge --------------------------------------------------------

    @classmethod
    def from_pgmpy(cls, network: DiscreteBayesianNetwork) -> "NetworkSpec":
        spec = cls()
        for cpd in network.get_cpds():
            var = cpd.variable
            states = tuple(cpd.state_names[var])
            parents = tuple(cpd.variables[1:])
            parent_states = [cpd.state_names[p] for p in parents]
            values = np.asarray(cpd.get_values(), dtype=float)
            cpt: CptTable = {}
            if parents:
                for col, combo in enumerate(product(*parent_states)):
                    cpt[tuple(combo)] = [float(x) for x in values[:, col]]
            else:
                cpt[()] = [float(x) for x in values[:, 0]]
            spec.add(DiscreteNode(name=var, states=states, parents=parents, cpt=cpt))
        return spec

    def to_pgmpy(self) -> DiscreteBayesianNetwork:
        self.validate()
        net = DiscreteBayesianNetwork(self.edges())
        # ensure isolated (parentless, childless) nodes are present too
        net.add_nodes_from(self.nodes.keys())
        all_states = self.states
        cpds = []
        for node in self.nodes.values():
            parent_states = [list(all_states[p]) for p in node.parents]
            columns = list(product(*parent_states)) if node.parents else [()]
            values = np.zeros((len(node.states), len(columns)))
            for col_idx, combo in enumerate(columns):
                values[:, col_idx] = node.cpt[tuple(combo)]
            state_names = {node.name: list(node.states)}
            for p, ps in zip(node.parents, parent_states):
                state_names[p] = ps
            cpds.append(
                TabularCPD(
                    variable=node.name,
                    variable_card=len(node.states),
                    values=values.tolist(),
                    evidence=list(node.parents) or None,
                    evidence_card=[len(s) for s in parent_states] or None,
                    state_names=state_names,
                )
            )
        net.add_cpds(*cpds)
        net.check_model()
        return net


__all__ = ["DiscreteNode", "NetworkSpec", "CptTable"]
