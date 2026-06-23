"""Bayesian network for the Strait of Hormuz scenario demo.

All conditional probability tables here are *elicited*, not learned from
data. The numbers are illustrative and chosen to be internally consistent
with three scenario narratives (Stress Mitigates / Prolonged Conflict /
Severe Closure). Each CPT carries a brief comment explaining the
qualitative reasoning behind the numbers.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Sequence, Tuple

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

# State vocabularies (single source of truth used everywhere).
STATES: Dict[str, List[str]] = {
    "US_Iran_Negotiations": ["success", "stalled", "breakdown"],
    "Iranian_Regime_Stability": ["stable", "pressured", "unstable"],
    "Third_Party_Mediation": ["active", "none"],
    "Sanctions_Trajectory": ["easing", "status_quo", "tightening"],
    "Iran_Aligned_Militia_Attacks": ["low", "elevated", "high"],
    "Tanker_Incidents": ["none", "isolated", "frequent"],
    "US_Military_Response": ["none", "limited", "major"],
    "Strait_Operationally_Closed": ["no", "partial", "full"],
    "Energy_Infrastructure_Damage": ["none", "moderate", "severe"],
    "Conflict_Duration": ["short", "medium", "long"],
    "Diplomatic_Resolution_Path": ["open", "narrowing", "closed"],
    "Oil_Price_Regime": ["below_90", "90_to_120", "above_120"],
    "Scenario": ["Stress_Mitigates", "Prolonged_Conflict", "Severe_Closure"],
}

EDGES: List[Tuple[str, str]] = [
    ("Iranian_Regime_Stability", "Iran_Aligned_Militia_Attacks"),
    ("Sanctions_Trajectory", "Iran_Aligned_Militia_Attacks"),
    ("Iran_Aligned_Militia_Attacks", "Tanker_Incidents"),
    ("US_Iran_Negotiations", "Tanker_Incidents"),
    ("Tanker_Incidents", "US_Military_Response"),
    ("Sanctions_Trajectory", "US_Military_Response"),
    ("Tanker_Incidents", "Strait_Operationally_Closed"),
    ("US_Military_Response", "Strait_Operationally_Closed"),
    ("US_Military_Response", "Energy_Infrastructure_Damage"),
    ("Strait_Operationally_Closed", "Energy_Infrastructure_Damage"),
    ("US_Iran_Negotiations", "Conflict_Duration"),
    ("Third_Party_Mediation", "Conflict_Duration"),
    ("US_Military_Response", "Conflict_Duration"),
    ("US_Iran_Negotiations", "Diplomatic_Resolution_Path"),
    ("Third_Party_Mediation", "Diplomatic_Resolution_Path"),
    ("Iranian_Regime_Stability", "Diplomatic_Resolution_Path"),
    ("Strait_Operationally_Closed", "Oil_Price_Regime"),
    ("Energy_Infrastructure_Damage", "Oil_Price_Regime"),
    ("Energy_Infrastructure_Damage", "Scenario"),
    ("Conflict_Duration", "Scenario"),
    ("Diplomatic_Resolution_Path", "Scenario"),
]

# Latent-regime topology (Plan 1): reverse the Scenario arrows and give S the
# downstream-most mediator parents Pa(S) = {US_Military_Response,
# Strait_Operationally_Closed}. Drop {D,T,P}->S; add S->{D,T,P}, M->S, C->S.
# The upstream chain and Oil_Price_Regime are structurally untouched.
_S = "Scenario"
_D, _T, _P = (
    "Energy_Infrastructure_Damage",
    "Conflict_Duration",
    "Diplomatic_Resolution_Path",
)
_M, _C = "US_Military_Response", "Strait_Operationally_Closed"
_S_REMOVE = {(_D, _S), (_T, _S), (_P, _S)}
_S_ADD = [(_S, _D), (_S, _T), (_S, _P), (_M, _S), (_C, _S)]
EDGES_LATENT: List[Tuple[str, str]] = [e for e in EDGES if e not in _S_REMOVE] + _S_ADD


def _cpd(
    var: str,
    parents: Sequence[str],
    table: Dict[Tuple[str, ...], Sequence[float]],
) -> TabularCPD:
    """Build a TabularCPD from a {parent-state-tuple: child-prob-vector} dict.

    Columns are emitted in the canonical pgmpy order (last parent varies
    fastest), so callers can write the table in any key order.
    """
    var_states = STATES[var]
    parent_states = [STATES[p] for p in parents]
    parent_cards = [len(s) for s in parent_states]
    columns = list(product(*parent_states)) if parents else [()]
    values: List[List[float]] = [[0.0] * len(columns) for _ in var_states]
    for col_idx, key in enumerate(columns):
        probs = table[key] if parents else table[()]
        if abs(sum(probs) - 1.0) > 1e-8:
            raise ValueError(f"CPT column for {var}{key} sums to {sum(probs)}")
        for row_idx, p in enumerate(probs):
            values[row_idx][col_idx] = p
    state_names = {var: var_states}
    for p, s in zip(parents, parent_states):
        state_names[p] = s
    return TabularCPD(
        variable=var,
        variable_card=len(var_states),
        values=values,
        evidence=list(parents) if parents else None,
        evidence_card=parent_cards if parents else None,
        state_names=state_names,
    )


# ---------------------------------------------------------------------------
# Root priors
# ---------------------------------------------------------------------------

# Reasoning: base rate sees most negotiations stalling, with breakdown a
# meaningful tail; outright success is the minority outcome.
CPD_NEGOTIATIONS = _cpd(
    "US_Iran_Negotiations", [], {(): [0.20, 0.55, 0.25]}
)

# Reasoning: regime is most often "pressured" under sanctions; outright
# instability is rarer but non-trivial.
CPD_REGIME = _cpd(
    "Iranian_Regime_Stability", [], {(): [0.30, 0.50, 0.20]}
)

# Reasoning: third-party mediators (Oman, Qatar, EU) are active roughly
# half the time during Gulf flare-ups.
CPD_MEDIATION = _cpd(
    "Third_Party_Mediation", [], {(): [0.45, 0.55]}
)

# Reasoning: status quo dominates; tightening more likely than easing in
# the current geopolitical environment.
CPD_SANCTIONS = _cpd(
    "Sanctions_Trajectory", [], {(): [0.15, 0.55, 0.30]}
)


# ---------------------------------------------------------------------------
# Iran_Aligned_Militia_Attacks | Regime_Stability, Sanctions_Trajectory
# Reasoning: stable regime + easing sanctions → low activity; unstable
# regime + tightening sanctions → high activity (incentive to lash out).
# ---------------------------------------------------------------------------
CPD_MILITIA = _cpd(
    "Iran_Aligned_Militia_Attacks",
    ["Iranian_Regime_Stability", "Sanctions_Trajectory"],
    {
        ("stable",   "easing"):      [0.85, 0.13, 0.02],
        ("stable",   "status_quo"):  [0.70, 0.25, 0.05],
        ("stable",   "tightening"):  [0.45, 0.40, 0.15],
        ("pressured","easing"):      [0.55, 0.35, 0.10],
        ("pressured","status_quo"):  [0.30, 0.50, 0.20],
        ("pressured","tightening"):  [0.10, 0.45, 0.45],
        ("unstable", "easing"):      [0.30, 0.45, 0.25],
        ("unstable", "status_quo"):  [0.15, 0.40, 0.45],
        ("unstable", "tightening"):  [0.05, 0.25, 0.70],
    },
)


# ---------------------------------------------------------------------------
# Tanker_Incidents | Iran_Aligned_Militia_Attacks, US_Iran_Negotiations
# Reasoning: incidents track militia attack tempo but are dampened by
# successful negotiations (back-channel restraint).
# ---------------------------------------------------------------------------
CPD_TANKERS = _cpd(
    "Tanker_Incidents",
    ["Iran_Aligned_Militia_Attacks", "US_Iran_Negotiations"],
    {
        ("low",      "success"):   [0.92, 0.07, 0.01],
        ("low",      "stalled"):   [0.85, 0.13, 0.02],
        ("low",      "breakdown"): [0.70, 0.25, 0.05],
        ("elevated", "success"):   [0.55, 0.40, 0.05],
        ("elevated", "stalled"):   [0.30, 0.55, 0.15],
        ("elevated", "breakdown"): [0.15, 0.55, 0.30],
        ("high",     "success"):   [0.20, 0.50, 0.30],
        ("high",     "stalled"):   [0.05, 0.45, 0.50],
        ("high",     "breakdown"): [0.02, 0.23, 0.75],
    },
)


# ---------------------------------------------------------------------------
# US_Military_Response | Tanker_Incidents, Sanctions_Trajectory
# Reasoning: tightening posture correlates with willingness to use force
# in response to incidents; isolated incidents rarely draw a major response.
# ---------------------------------------------------------------------------
CPD_MILITARY = _cpd(
    "US_Military_Response",
    ["Tanker_Incidents", "Sanctions_Trajectory"],
    {
        ("none",     "easing"):     [0.95, 0.04, 0.01],
        ("none",     "status_quo"): [0.90, 0.09, 0.01],
        ("none",     "tightening"): [0.80, 0.18, 0.02],
        ("isolated", "easing"):     [0.60, 0.37, 0.03],
        ("isolated", "status_quo"): [0.40, 0.52, 0.08],
        ("isolated", "tightening"): [0.20, 0.60, 0.20],
        ("frequent", "easing"):     [0.20, 0.55, 0.25],
        ("frequent", "status_quo"): [0.08, 0.52, 0.40],
        ("frequent", "tightening"): [0.02, 0.33, 0.65],
    },
)


# ---------------------------------------------------------------------------
# Strait_Operationally_Closed | Tanker_Incidents, US_Military_Response
# Reasoning: closure follows either Iranian denial-of-passage (frequent
# incidents) or kinetic exchanges that drive shippers to pause transit.
# ---------------------------------------------------------------------------
CPD_STRAIT = _cpd(
    "Strait_Operationally_Closed",
    ["Tanker_Incidents", "US_Military_Response"],
    {
        ("none",     "none"):    [0.97, 0.025, 0.005],
        ("none",     "limited"): [0.85, 0.13,  0.02],
        ("none",     "major"):   [0.55, 0.35,  0.10],
        ("isolated", "none"):    [0.80, 0.17,  0.03],
        ("isolated", "limited"): [0.55, 0.35,  0.10],
        ("isolated", "major"):   [0.25, 0.50,  0.25],
        ("frequent", "none"):    [0.40, 0.45,  0.15],
        ("frequent", "limited"): [0.15, 0.55,  0.30],
        ("frequent", "major"):   [0.03, 0.37,  0.60],
    },
)


# ---------------------------------------------------------------------------
# Energy_Infrastructure_Damage | US_Military_Response, Strait_Closed
# Reasoning: damage requires kinetic action; full closure is often the
# *result* of damage to terminals/refineries on either side.
# ---------------------------------------------------------------------------
CPD_DAMAGE = _cpd(
    "Energy_Infrastructure_Damage",
    ["US_Military_Response", "Strait_Operationally_Closed"],
    {
        ("none",    "no"):      [0.97, 0.025, 0.005],
        ("none",    "partial"): [0.80, 0.17,  0.03],
        ("none",    "full"):    [0.50, 0.40,  0.10],
        ("limited", "no"):      [0.80, 0.18,  0.02],
        ("limited", "partial"): [0.50, 0.40,  0.10],
        ("limited", "full"):    [0.20, 0.55,  0.25],
        ("major",   "no"):      [0.40, 0.45,  0.15],
        ("major",   "partial"): [0.15, 0.55,  0.30],
        ("major",   "full"):    [0.03, 0.32,  0.65],
    },
)


# ---------------------------------------------------------------------------
# Conflict_Duration | Negotiations, Mediation, US_Military_Response
# Reasoning: success + active mediation → short; breakdown + major
# response → long. 3 x 2 x 3 = 18 columns.
# ---------------------------------------------------------------------------
CPD_DURATION = _cpd(
    "Conflict_Duration",
    ["US_Iran_Negotiations", "Third_Party_Mediation", "US_Military_Response"],
    {
        ("success",   "active", "none"):    [0.90, 0.08, 0.02],
        ("success",   "active", "limited"): [0.70, 0.25, 0.05],
        ("success",   "active", "major"):   [0.40, 0.45, 0.15],
        ("success",   "none",   "none"):    [0.80, 0.17, 0.03],
        ("success",   "none",   "limited"): [0.55, 0.35, 0.10],
        ("success",   "none",   "major"):   [0.25, 0.50, 0.25],
        ("stalled",   "active", "none"):    [0.55, 0.35, 0.10],
        ("stalled",   "active", "limited"): [0.30, 0.50, 0.20],
        ("stalled",   "active", "major"):   [0.10, 0.45, 0.45],
        ("stalled",   "none",   "none"):    [0.40, 0.45, 0.15],
        ("stalled",   "none",   "limited"): [0.15, 0.50, 0.35],
        ("stalled",   "none",   "major"):   [0.05, 0.30, 0.65],
        ("breakdown", "active", "none"):    [0.25, 0.50, 0.25],
        ("breakdown", "active", "limited"): [0.10, 0.45, 0.45],
        ("breakdown", "active", "major"):   [0.03, 0.27, 0.70],
        ("breakdown", "none",   "none"):    [0.15, 0.45, 0.40],
        ("breakdown", "none",   "limited"): [0.05, 0.30, 0.65],
        ("breakdown", "none",   "major"):   [0.02, 0.13, 0.85],
    },
)


# ---------------------------------------------------------------------------
# Diplomatic_Resolution_Path | Negotiations, Mediation, Regime_Stability
# Reasoning: open path needs willing counterparties + a stable regime to
# enforce any deal; an unstable regime collapses the path even when
# negotiations nominally succeed. 3 x 2 x 3 = 18 columns.
# ---------------------------------------------------------------------------
CPD_DIPLO = _cpd(
    "Diplomatic_Resolution_Path",
    ["US_Iran_Negotiations", "Third_Party_Mediation", "Iranian_Regime_Stability"],
    {
        ("success",   "active", "stable"):     [0.92, 0.07, 0.01],
        ("success",   "active", "pressured"):  [0.80, 0.17, 0.03],
        ("success",   "active", "unstable"):   [0.45, 0.40, 0.15],
        ("success",   "none",   "stable"):     [0.80, 0.17, 0.03],
        ("success",   "none",   "pressured"):  [0.65, 0.28, 0.07],
        ("success",   "none",   "unstable"):   [0.30, 0.45, 0.25],
        ("stalled",   "active", "stable"):     [0.55, 0.35, 0.10],
        ("stalled",   "active", "pressured"):  [0.40, 0.45, 0.15],
        ("stalled",   "active", "unstable"):   [0.15, 0.45, 0.40],
        ("stalled",   "none",   "stable"):     [0.35, 0.50, 0.15],
        ("stalled",   "none",   "pressured"):  [0.20, 0.50, 0.30],
        ("stalled",   "none",   "unstable"):   [0.05, 0.35, 0.60],
        ("breakdown", "active", "stable"):     [0.20, 0.50, 0.30],
        ("breakdown", "active", "pressured"):  [0.10, 0.45, 0.45],
        ("breakdown", "active", "unstable"):   [0.03, 0.27, 0.70],
        ("breakdown", "none",   "stable"):     [0.10, 0.40, 0.50],
        ("breakdown", "none",   "pressured"):  [0.04, 0.26, 0.70],
        ("breakdown", "none",   "unstable"):   [0.01, 0.09, 0.90],
    },
)


# ---------------------------------------------------------------------------
# Oil_Price_Regime | Strait_Closed, Energy_Damage
# Reasoning: full closure plus severe damage prices in supply destruction;
# no closure / no damage keeps prices in the lower band.
# ---------------------------------------------------------------------------
CPD_OIL = _cpd(
    "Oil_Price_Regime",
    ["Strait_Operationally_Closed", "Energy_Infrastructure_Damage"],
    {
        ("no",      "none"):     [0.75, 0.22, 0.03],
        ("no",      "moderate"): [0.40, 0.50, 0.10],
        ("no",      "severe"):   [0.15, 0.55, 0.30],
        ("partial", "none"):     [0.30, 0.55, 0.15],
        ("partial", "moderate"): [0.10, 0.55, 0.35],
        ("partial", "severe"):   [0.03, 0.32, 0.65],
        ("full",    "none"):     [0.10, 0.50, 0.40],
        ("full",    "moderate"): [0.03, 0.32, 0.65],
        ("full",    "severe"):   [0.01, 0.09, 0.90],
    },
)


# ---------------------------------------------------------------------------
# Scenario | Energy_Damage, Conflict_Duration, Diplomatic_Path
# Reasoning: this CPT structurally encodes the three client scenarios.
#   Stress_Mitigates  : low/no damage, short/medium duration, open path
#   Prolonged_Conflict: moderate damage, long duration, narrowing/closed path
#   Severe_Closure    : severe damage, long duration, closed path
# 3 x 3 x 3 = 27 columns.
# ---------------------------------------------------------------------------
CPD_SCENARIO = _cpd(
    "Scenario",
    ["Energy_Infrastructure_Damage", "Conflict_Duration", "Diplomatic_Resolution_Path"],
    {
        # ---- damage = none ----
        ("none",     "short",  "open"):      [0.94, 0.05, 0.01],
        ("none",     "short",  "narrowing"): [0.85, 0.13, 0.02],
        ("none",     "short",  "closed"):    [0.65, 0.30, 0.05],
        ("none",     "medium", "open"):      [0.85, 0.13, 0.02],
        ("none",     "medium", "narrowing"): [0.65, 0.30, 0.05],
        ("none",     "medium", "closed"):    [0.40, 0.50, 0.10],
        ("none",     "long",   "open"):      [0.55, 0.40, 0.05],
        ("none",     "long",   "narrowing"): [0.30, 0.60, 0.10],
        ("none",     "long",   "closed"):    [0.15, 0.65, 0.20],
        # ---- damage = moderate ----
        ("moderate", "short",  "open"):      [0.70, 0.25, 0.05],
        ("moderate", "short",  "narrowing"): [0.50, 0.40, 0.10],
        ("moderate", "short",  "closed"):    [0.30, 0.50, 0.20],
        ("moderate", "medium", "open"):      [0.45, 0.45, 0.10],
        ("moderate", "medium", "narrowing"): [0.20, 0.60, 0.20],
        ("moderate", "medium", "closed"):    [0.10, 0.60, 0.30],
        ("moderate", "long",   "open"):      [0.20, 0.65, 0.15],
        ("moderate", "long",   "narrowing"): [0.07, 0.68, 0.25],
        ("moderate", "long",   "closed"):    [0.03, 0.55, 0.42],
        # ---- damage = severe ----
        ("severe",   "short",  "open"):      [0.30, 0.45, 0.25],
        ("severe",   "short",  "narrowing"): [0.15, 0.50, 0.35],
        ("severe",   "short",  "closed"):    [0.07, 0.43, 0.50],
        ("severe",   "medium", "open"):      [0.15, 0.50, 0.35],
        ("severe",   "medium", "narrowing"): [0.05, 0.45, 0.50],
        ("severe",   "medium", "closed"):    [0.02, 0.28, 0.70],
        ("severe",   "long",   "open"):      [0.05, 0.40, 0.55],
        ("severe",   "long",   "narrowing"): [0.02, 0.23, 0.75],
        ("severe",   "long",   "closed"):    [0.01, 0.09, 0.90],
    },
)


SCENARIO_NARRATIVES: Dict[str, str] = {
    "Stress_Mitigates": (
        "No near-term resolution, but the stress itself spurs an "
        "agreement. Lasting impacts are mitigated."
    ),
    "Prolonged_Conflict": (
        "Iran prolongs the conflict; the strait stays closed past April. "
        "Supply-chain impacts propagate non-linearly."
    ),
    "Severe_Closure": (
        "Strait remains closed; material damage to energy infrastructure. "
        "Severe growth shock plus medium-term inflation pricing."
    ),
}

# Modal (most-probable) outcome signature each regime generates, written
# (Energy_Infrastructure_Damage, Conflict_Duration, Diplomatic_Resolution_Path).
# Under the latent-regime reading (Plan §A.2.4) these are the *modes* of each regime's
# emission distribution, not definitions of it. Used as the narrative-mode invariant in
# validation. NOTE: the v1 bootstrap reproduces every extreme-scenario mode; only
# Prolonged_Conflict's duration mode lands on 'medium' rather than 'long' (the documented
# poorly-identified middle regime), pending Plan 4 elicitation.
SCENARIO_SIGNATURES: Dict[str, Tuple[str, str, str]] = {
    "Stress_Mitigates": ("none", "short", "open"),
    "Prolonged_Conflict": ("moderate", "long", "narrowing"),
    "Severe_Closure": ("severe", "long", "closed"),
}

# Labelling-topology CPDs in add order (Scenario is a leaf).
_LABELLING_CPDS = (
    CPD_NEGOTIATIONS, CPD_REGIME, CPD_MEDIATION, CPD_SANCTIONS, CPD_MILITIA,
    CPD_TANKERS, CPD_MILITARY, CPD_STRAIT, CPD_DAMAGE, CPD_DURATION, CPD_DIPLO,
    CPD_OIL, CPD_SCENARIO,
)
# Upstream + Oil CPDs that carry over UNCHANGED into the latent topology (the labelling
# outcome CPDs CPD_DAMAGE/CPD_DURATION/CPD_DIPLO and CPD_SCENARIO are replaced there).
_LATENT_SHARED_CPDS = (
    CPD_NEGOTIATIONS, CPD_REGIME, CPD_MEDIATION, CPD_SANCTIONS, CPD_MILITIA,
    CPD_TANKERS, CPD_MILITARY, CPD_STRAIT, CPD_OIL,
)


def build_network(topology: str = "labelling") -> DiscreteBayesianNetwork:
    """Assemble and return the fully-specified Bayesian network.

    ``topology="labelling"`` (default) is the original model: ``Scenario`` is a leaf
    classified by the outcomes ``P(S | D, T, P)``. ``topology="latent_regime"`` is the
    Plan 1 reframe: ``Scenario`` is a latent regime generating the outcomes, with the
    four anchor-derived CPTs from :mod:`src.cpt_data`.
    """
    if topology == "labelling":
        net = DiscreteBayesianNetwork(EDGES)
        net.add_cpds(*_LABELLING_CPDS)
    elif topology == "latent_regime":
        from .cpt_data import LATENT_CPTS, LATENT_CPT_PARENTS  # generated; lazy import
        net = DiscreteBayesianNetwork(EDGES_LATENT)
        emitted = [
            _cpd(child, LATENT_CPT_PARENTS[child], LATENT_CPTS[child])
            for child in LATENT_CPT_PARENTS
        ]
        net.add_cpds(*_LATENT_SHARED_CPDS, *emitted)
    else:
        raise ValueError(
            f"Unknown topology {topology!r}; expected 'labelling' or 'latent_regime'."
        )
    net.check_model()
    return net


__all__ = [
    "STATES",
    "EDGES",
    "EDGES_LATENT",
    "SCENARIO_NARRATIVES",
    "SCENARIO_SIGNATURES",
    "build_network",
]
