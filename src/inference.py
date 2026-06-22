"""Inference wrapper for the active scenario pack's Bayesian network."""
# TODO(pack-separation): docstring previously named the Hormuz scenario; the module
# is pack-agnostic (reads via src.scenario) — wording fixed, no Hormuz logic remains.

from __future__ import annotations

from typing import Dict, Mapping, Optional

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from src.scenario import LATENT, STATES, build_network

SCENARIO = LATENT  # the latent regime node of the active pack


def _require_latent_regime(network: DiscreteBayesianNetwork) -> None:
    """Raise unless ``network`` is the latent-regime topology.

    Regime Bayes factors are only meaningful when ``Scenario`` is a latent cause
    that *generates* the outcomes (i.e. has children). On the labelling topology
    ``Scenario`` is a leaf, so this query would silently return plausible-looking
    but meaningless numbers — fail loudly instead. Note this guards the regime
    *Bayes-factor* path only; :func:`clamped_scenario_likelihoods` is a
    deliberately topology-agnostic primitive (used on both topologies to
    contrast their likelihoods) and is intentionally left unguarded.
    """
    if not list(network.successors(SCENARIO)):
        raise ValueError(
            "scenario_bayes_factors requires the latent-regime topology "
            "(build_network('latent_regime')); on the labelling topology "
            f"{SCENARIO!r} is a leaf and regime Bayes factors are undefined."
        )


def _scenario_virtual_cpds(
    soft_evidence: Optional[Mapping[str, Mapping[str, float]]],
) -> list[TabularCPD]:
    cpds: list[TabularCPD] = []
    for node, dist in (soft_evidence or {}).items():
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


def scenario_bayes_factors(
    network: DiscreteBayesianNetwork,
    evidence: Optional[Mapping[str, str]] = None,
    soft_evidence: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, object]:
    """Regime Bayes factors for the latent-regime topology.

    Returns a dict with the regime ``prior`` ``P(S)``, the ``posterior`` ``P(S | E)``,
    the relative likelihoods ``rel_like`` (normalised, proportional to ``P(E | S=s)``),
    and the pairwise ``lambda`` matrix ``Lambda[s1][s2] = P(E|s1)/P(E|s2)``.

    Uses the ratio identity ``Lambda = [P(s1|E)/P(s1)] / [P(s2|E)/P(s2)]`` (the ``P(E)``
    cancels), which is exact for hard *and* soft evidence. For an algebraically
    independent cross-check on hard evidence see :func:`clamped_scenario_likelihoods`.
    """
    _require_latent_regime(network)
    evidence = dict(evidence or {})
    ve = VariableElimination(network)

    pri = ve.query([SCENARIO], evidence={}, show_progress=False)
    prior = {s: float(pri.get_value(**{SCENARIO: s})) for s in STATES[SCENARIO]}

    kw: Dict[str, object] = {"evidence": evidence, "show_progress": False}
    vcpds = _scenario_virtual_cpds(soft_evidence)
    if vcpds:
        kw["virtual_evidence"] = vcpds
    pos = ve.query([SCENARIO], **kw)
    posterior = {s: float(pos.get_value(**{SCENARIO: s})) for s in STATES[SCENARIO]}

    lr = {
        s: (posterior[s] / prior[s] if prior[s] > 0 else float("inf"))
        for s in STATES[SCENARIO]
    }
    z = sum(v for v in lr.values() if v != float("inf")) or 1.0
    rel_like = {s: (lr[s] / z if lr[s] != float("inf") else float("inf")) for s in lr}
    lam = {
        s1: {
            s2: (lr[s1] / lr[s2] if lr[s2] not in (0.0, float("inf")) else float("inf"))
            for s2 in STATES[SCENARIO]
        }
        for s1 in STATES[SCENARIO]
    }
    return {"prior": prior, "posterior": posterior, "rel_like": rel_like, "lambda": lam}


def clamped_scenario_likelihoods(
    network: DiscreteBayesianNetwork, evidence: Mapping[str, str]
) -> Dict[str, float]:
    """``P(E=e | S=s)`` via the three-clamped-inferences pattern (Plan §B.1 item 3).

    For each scenario state, clamp ``S=s`` and read the joint probability of the observed
    evidence configuration off ``P(evidence vars | S=s)``. Hard evidence only; gives the
    absolute per-regime likelihood (an independent check on the ratio method's factors).
    """
    if not evidence:
        raise ValueError("clamped_scenario_likelihoods requires non-empty hard evidence")
    ve = VariableElimination(network)
    evars = list(evidence.keys())
    out: Dict[str, float] = {}
    for s in STATES[SCENARIO]:
        joint = ve.query(evars, evidence={SCENARIO: s}, show_progress=False)
        out[s] = float(joint.get_value(**dict(evidence)))
    return out


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
        """Store per-node soft evidence as **likelihood ratios** (A1 semantics).

        Values are normalised by their max (the best-supported state → 1.0), not
        by their sum: pgmpy's virtual-evidence treats the column as a likelihood
        and applies it proportionally, so max-pinning is mathematically
        equivalent to sum-normalising for inference while keeping the stored
        values interpretable as the ε vector the translator emitted.
        """
        for node, dist in soft_evidence.items():
            if node not in STATES:
                raise KeyError(f"Unknown node: {node}")
            probs: Dict[str, float] = {}
            for state in STATES[node]:
                p = float(dist.get(state, 0.0))
                if p < 0.0:
                    raise ValueError(f"Negative likelihood for {node}.{state}: {p}")
                probs[state] = p
            peak = max(probs.values())
            if peak <= 0.0:
                raise ValueError(f"Soft evidence for {node} is all-zero.")
            probs = {k: v / peak for k, v in probs.items()}
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
        result = self._engine.query([SCENARIO], evidence={}, show_progress=False)
        return self._distribution(result, SCENARIO)

    def get_scenario_probabilities(self) -> Dict[str, float]:
        """Scenario marginal under the current accumulated evidence.

        Scenario is the query target, so any (invalid) evidence on it is dropped
        — mirroring :meth:`get_node_marginal` — rather than letting pgmpy reject
        the query.
        """
        ev = {k: v for k, v in self._evidence.items() if k != SCENARIO}
        result = self._engine.query(
            [SCENARIO],
            evidence=ev,
            virtual_evidence=self._virtual_evidence_cpds(),
            show_progress=False,
        )
        return self._distribution(result, SCENARIO)

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

    def scenario_bayes_factors(self) -> Dict[str, object]:
        """Regime Bayes factors under the engine's current evidence.

        Only meaningful on a latent-regime network; see
        :func:`scenario_bayes_factors` for the contract.
        """
        return scenario_bayes_factors(
            self._network, self._evidence, self._soft_evidence
        )

    def standalone_bayes_factors(
        self,
        evidence: Mapping[str, str],
        soft_evidence: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> Dict[str, object]:
        """Regime Bayes factors for a *specific* evidence set (not the engine's
        accumulated one) — used to attribute a single observation's contribution
        to the latent regime. Raises on a non-latent-regime network.
        """
        return scenario_bayes_factors(self._network, evidence, soft_evidence)

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


__all__ = [
    "BNInferenceEngine",
    "scenario_bayes_factors",
    "clamped_scenario_likelihoods",
]
