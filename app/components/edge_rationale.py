"""Edge-rationale view (Plan 5 P4). Self-contained: the per-topology edge
rationale + omission tables and the render function. Extracted from the dashboard.
"""
from __future__ import annotations


_EDGE_RATIONALE: list[tuple[str, str, str]] = [
    ("Iranian_Regime_Stability", "Iran_Aligned_Militia_Attacks",
     "An unstable or pressured regime has stronger incentive to lash out "
     "externally via its militia network; a stable regime can afford "
     "restraint and tighter command over proxies."),
    ("Sanctions_Trajectory", "Iran_Aligned_Militia_Attacks",
     "Tightening sanctions remove peaceful off-ramps and push Tehran to "
     "impose cost asymmetrically through proxies; easing sanctions "
     "reward restraint."),
    ("Iran_Aligned_Militia_Attacks", "Tanker_Incidents",
     "Houthi drone/missile strikes and IRGC-linked harassment are the "
     "direct mechanism behind most attacks on Gulf and Red Sea shipping."),
    ("US_Iran_Negotiations", "Tanker_Incidents",
     "Active back-channels dampen incidents via restraint signals; "
     "negotiation breakdowns remove that brake."),
    ("Tanker_Incidents", "US_Military_Response",
     "Observable kinetic events on shipping (strikes, seizures) are the "
     "visible trigger for carrier redeployments, escorts, and retaliatory "
     "strikes."),
    ("Sanctions_Trajectory", "US_Military_Response",
     "A hawkish sanctions posture correlates with political willingness "
     "to use force; an easing posture correlates with restraint."),
    ("Tanker_Incidents", "Strait_Operationally_Closed",
     "Insurance premiums, charterer avoidance, and convoy logistics mean "
     "enough incidents produce de facto closure even without Iranian "
     "mining."),
    ("US_Military_Response", "Strait_Operationally_Closed",
     "A major US response can either re-open traffic via escorts or "
     "provoke Iranian mining/closure attempts — both captured as "
     "dependence."),
    ("US_Military_Response", "Energy_Infrastructure_Damage",
     "Strikes on IRGC naval assets or Iranian oil facilities — and "
     "Iranian retaliation on Saudi/UAE/US infrastructure — are driven by "
     "the intensity of the military response."),
    ("Strait_Operationally_Closed", "Energy_Infrastructure_Damage",
     "Extended closure correlates with the broader escalation regime "
     "that brings production and export infrastructure into the "
     "crosshairs."),
    ("US_Iran_Negotiations", "Conflict_Duration",
     "Active direct talks accelerate resolution; breakdowns prolong the "
     "crisis by removing the obvious exit."),
    ("Third_Party_Mediation", "Conflict_Duration",
     "Qatari, Omani, or Chinese mediation compresses timelines to a "
     "deal by providing face-saving channels."),
    ("US_Military_Response", "Conflict_Duration",
     "A major response commits the US to a protracted campaign; no "
     "response allows quicker de-escalation."),
    ("US_Iran_Negotiations", "Diplomatic_Resolution_Path",
     "The status of direct bilateral talks is the primary determinant "
     "of whether a resolution path stays open."),
    ("Third_Party_Mediation", "Diplomatic_Resolution_Path",
     "External facilitators open or preserve channels precisely when "
     "direct talks stall."),
    ("Iranian_Regime_Stability", "Diplomatic_Resolution_Path",
     "A stable regime can credibly commit and bind hardliner factions; "
     "an unstable regime cannot deliver on any deal it signs."),
    ("Strait_Operationally_Closed", "Oil_Price_Regime",
     "Roughly 20% of seaborne oil transits Hormuz — closure is the "
     "single largest supply-shock lever in the model."),
    ("Energy_Infrastructure_Damage", "Oil_Price_Regime",
     "Damaged production or export capacity removes barrels from the "
     "market for quarters, not days."),
    ("Energy_Infrastructure_Damage", "Scenario",
     "Severe infrastructure damage pushes the scenario toward "
     "Prolonged_Conflict or Severe_Closure."),
    ("Conflict_Duration", "Scenario",
     "A long conflict precludes Stress_Mitigates and favors the two "
     "heavier scenarios."),
    ("Diplomatic_Resolution_Path", "Scenario",
     "An open resolution path is the main driver of the "
     "Stress_Mitigates outcome."),
]

_EDGE_OMISSIONS: list[tuple[str, str, str]] = [
    ("Iran_Aligned_Militia_Attacks", "US_Military_Response",
     "Assumed mediated by Tanker_Incidents: Washington reacts to "
     "observable kinetic events, not militia posture in the abstract. "
     "Known limitation — US base attacks in Iraq/Syria have historically "
     "triggered direct US strikes without any tanker incident."),
    ("Iranian_Regime_Stability", "Tanker_Incidents",
     "Mediated by Iran_Aligned_Militia_Attacks: regime posture reaches "
     "the water only through the proxy network."),
    ("Sanctions_Trajectory", "Tanker_Incidents",
     "Mediated by Iran_Aligned_Militia_Attacks — same reasoning. The "
     "direct sanctions→incidents arrow is absorbed into the militia "
     "channel."),
    ("US_Military_Response", "Oil_Price_Regime",
     "Mediated by Strait_Operationally_Closed and "
     "Energy_Infrastructure_Damage. The oil market prices physical "
     "flows and capacity, not force posture directly."),
    ("Strait_Operationally_Closed", "Scenario",
     "Mediated by Oil_Price_Regime and Energy_Infrastructure_Damage. "
     "The Scenario node classifies on downstream outcomes, not on the "
     "mechanism that produced them."),
    ("Third_Party_Mediation", "Iran_Aligned_Militia_Attacks",
     "Mediation is modeled as operating on negotiations and duration, "
     "not on kinetic tempo. Debatable — Oman has historically passed "
     "de-escalation messages to the IRGC during flashpoints."),
    ("Iranian_Regime_Stability", "US_Military_Response",
     "Omitted for parsimony: US response is driven by events plus "
     "sanctions posture in the model, not by Iranian internal politics "
     "directly."),
]

# --- Latent-regime topology (Plan 1) -----------------------------------------
# The three {D,T,P} -> Scenario classifier edges reverse into emissions, and
# Scenario gains the context parents {US_Military_Response, Strait_Operationally_Closed}.
_LATENT_SCENARIO_EDGES: list[tuple[str, str, str]] = [
    ("US_Military_Response", "Scenario",
     "Context parent of the regime: a major US military response raises the prior "
     "for the Severe/Prolonged regimes before any outcome is observed."),
    ("Strait_Operationally_Closed", "Scenario",
     "Context parent of the regime: observed strait closure is a strong prior signal "
     "for the Severe regime. Retained alongside military response (partly redundant) "
     "for closure-evidence sensitivity — a deliberate parsimony exception."),
    ("Scenario", "Energy_Infrastructure_Damage",
     "Emission: the regime generates damage. The Severe regime concentrates mass on "
     "severe damage but keeps nonzero off-mode mass (overlap is by design)."),
    ("Scenario", "Conflict_Duration",
     "Emission: the regime generates conflict duration (Prolonged/Severe lean long)."),
    ("Scenario", "Diplomatic_Resolution_Path",
     "Emission: the regime generates the diplomatic path (Stress leans open, "
     "Severe leans closed)."),
]
_EDGE_RATIONALE_LATENT: list[tuple[str, str, str]] = (
    [e for e in _EDGE_RATIONALE if e[1] != "Scenario"] + _LATENT_SCENARIO_EDGES
)
_EDGE_OMISSIONS_LATENT: list[tuple[str, str, str]] = (
    [e for e in _EDGE_OMISSIONS if e[1] != "Scenario"] + [
        ("Third_Party_Mediation", "Scenario",
         "Documented blind spot (Plan 1 §A.4): mediation has no direct path to the "
         "regime in v1 — it reaches Scenario only indirectly, through the diplomatic-"
         "path and duration emissions. Accepted for v1; candidate direct parent in "
         "Plan 4."),
    ]
)

_RATIONALE_BY_TOPOLOGY = {
    "labelling": (_EDGE_RATIONALE, _EDGE_OMISSIONS),
    "latent_regime": (_EDGE_RATIONALE_LATENT, _EDGE_OMISSIONS_LATENT),
}


def _fmt_node(name: str) -> str:
    return name.replace("Iran_Aligned", "Iran-Aligned").replace("_", " ")


def edge_title_map(topology) -> dict:
    """``{(parent, child): rationale}`` for the *present* edges of a topology —
    used to populate the agraph edge hover tooltips (Plan 5 P10 / C11 / V13).
    Omitted edges are not in the graph, so they are not included.
    """
    rationale, _omissions = _RATIONALE_BY_TOPOLOGY[topology]
    return {(parent, child): reason for parent, child, reason in rationale}


def render(st, topology):
    st.markdown(
        "<div class='card-title'>Why each arrow is (or isn't) in the "
        "network</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Short rationales for every edge in the DAG, plus a list of "
        "plausible connections that were deliberately left out and why. "
        "These are modeling judgments, not settled truths — treat them "
        "as the assumptions behind the current CPTs."
    )

    _rationale, _omissions = _RATIONALE_BY_TOPOLOGY[topology]

    st.markdown("#### Edges present in the model")
    for parent, child, reason in _rationale:
        st.markdown(
            f"**{_fmt_node(parent)}** → **{_fmt_node(child)}**  \n"
            f"<span style='color:#475569'>{reason}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Notable omitted edges")
    st.caption(
        "Connections a domain reader might expect but that are not in "
        "the DAG — either because they're mediated by another node or "
        "because they were dropped for parsimony."
    )
    for parent, child, reason in _omissions:
        st.markdown(
            f"**{_fmt_node(parent)}** ⇢ **{_fmt_node(child)}** "
            f"<span style='color:#b45309'>(omitted)</span>  \n"
            f"<span style='color:#475569'>{reason}</span>",
            unsafe_allow_html=True,
        )
