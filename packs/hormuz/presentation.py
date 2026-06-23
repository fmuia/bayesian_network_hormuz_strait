"""Hormuz presentation data (moved out of app/theme.py, src/viz.py,
app/components/edge_rationale.py, app/components/model_explainer.py,
app/components/audit_view.py, src/translator_pipeline.py).

All display-only / domain content; the shared views read it via the
``src.scenario`` seam (``PRESENTATION``).
"""
from __future__ import annotations

from packs.base import Presentation
from packs.hormuz.network import SCENARIO_NARRATIVES

# --- latent-state colours / labels -----------------------------------------
SCENARIO_COLOR = {
    "Stress_Mitigates": "#2E8B57",
    "Prolonged_Conflict": "#D4A017",
    "Severe_Closure": "#B22222",
}
SCENARIO_LABEL = {
    "Stress_Mitigates": "Stress Mitigates",
    "Prolonged_Conflict": "Prolonged Conflict",
    "Severe_Closure": "Severe Closure",
}

# --- root-driver styling ----------------------------------------------------
ROOT_DRIVER_STYLE = {   # (background, foreground) — used by the dashboard chips
    "US_Iran_Negotiations": ("#DBEAFE", "#1D4ED8"),
    "Iranian_Regime_Stability": ("#FCE7F3", "#BE185D"),
    "Third_Party_Mediation": ("#FEF3C7", "#B45309"),
    "Sanctions_Trajectory": ("#EDE9FE", "#6D28D9"),
}
ROOT_DRIVER_COLORS = {  # (background, border, observed_fill) — used by the DAG
    "US_Iran_Negotiations": ("#DBEAFE", "#1D4ED8", "#1E40AF"),
    "Iranian_Regime_Stability": ("#FCE7F3", "#BE185D", "#9D174D"),
    "Third_Party_Mediation": ("#FEF3C7", "#B45309", "#92400E"),
    "Sanctions_Trajectory": ("#EDE9FE", "#6D28D9", "#5B21B6"),
}

# --- audit-view intermediate-node ordering ---------------------------------
INTERMEDIATE_NODES = [
    "Iran_Aligned_Militia_Attacks", "Tanker_Incidents", "US_Military_Response",
    "Strait_Operationally_Closed", "Energy_Infrastructure_Damage",
    "Conflict_Duration", "Diplomatic_Resolution_Path", "Oil_Price_Regime",
]

# --- deterministic offline (fake) translator keyword map -------------------
FALLBACK_KEYWORD_MAP = [
    (("tanker", "vessel", "shipping"), "Tanker_Incidents", "frequent"),
    (("militia",), "Iran_Aligned_Militia_Attacks", "elevated"),
    (("sanction",), "Sanctions_Trajectory", "easing"),
    (("back-channel", "negotiat", "talks"), "US_Iran_Negotiations", "stalled"),
    (("mediat", "oman", "qatar"), "Third_Party_Mediation", "active"),
    (("strait", "closure", "closed", "inspection"), "Strait_Operationally_Closed", "partial"),
    (("strike", "military", "irgc"), "US_Military_Response", "major"),
    (("missile", "fire", "terminal", "refinery", "damage"), "Energy_Infrastructure_Damage", "severe"),
    (("protest", "regime", "crackdown"), "Iranian_Regime_Stability", "pressured"),
    (("oil", "brent", "crude", "price"), "Oil_Price_Regime", "above_120"),
]

# --- edge rationale / omission tables (per topology) ------------------------
_EDGE_RATIONALE = [
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

_EDGE_OMISSIONS = [
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

_LATENT_SCENARIO_EDGES = [
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
_EDGE_RATIONALE_LATENT = (
    [e for e in _EDGE_RATIONALE if e[1] != "Scenario"] + _LATENT_SCENARIO_EDGES
)
_EDGE_OMISSIONS_LATENT = (
    [e for e in _EDGE_OMISSIONS if e[1] != "Scenario"] + [
        ("Third_Party_Mediation", "Scenario",
         "Documented blind spot (Plan 1 §A.4): mediation has no direct path to the "
         "regime in v1 — it reaches Scenario only indirectly, through the diplomatic-"
         "path and duration emissions. Accepted for v1; candidate direct parent in "
         "Plan 4."),
    ]
)

EDGE_RATIONALES = {"labelling": _EDGE_RATIONALE, "latent_regime": _EDGE_RATIONALE_LATENT}
EDGE_OMISSIONS = {"labelling": _EDGE_OMISSIONS, "latent_regime": _EDGE_OMISSIONS_LATENT}

# --- model-overview copy (per topology) ------------------------------------
def _overview_html(scenario_clause: str) -> str:
    return (
        "<div class='explain'>"
        "<p>The Bayesian network encodes qualitative causal structure "
        "between four <b>root drivers</b> (negotiations, regime "
        "stability, third-party mediation, sanctions trajectory), "
        "<b>eight intermediate nodes</b> (Iran-aligned militia attacks, tanker "
        "incidents, US military response, strait closure, energy "
        "infrastructure damage, conflict duration, diplomatic path, "
        f"oil price regime), and {scenario_clause} "
        "whose three states correspond to the client's strategic "
        "scenarios.</p>"
        "<h4>Two layers</h4>"
        "<p>A free-text headline is passed through an LLM translator "
        "that extracts BN-relevant probabilistic assignments (e.g. "
        "<i>\"Fourth tanker incident in two weeks\"</i> gives a high "
        "probability to <code>Tanker_Incidents = frequent</code>). "
        "Those soft assignments become BN evidence; variable-elimination "
        "propagates them and yields the posterior distribution at "
        "every node.</p>"
        "<h4>Scenario definitions</h4>"
        "<ul>"
        f"<li><b style='color:{SCENARIO_COLOR['Stress_Mitigates']};'>Stress Mitigates</b> — "
        f"{SCENARIO_NARRATIVES['Stress_Mitigates']}</li>"
        f"<li><b style='color:{SCENARIO_COLOR['Prolonged_Conflict']};'>Prolonged Conflict</b> — "
        f"{SCENARIO_NARRATIVES['Prolonged_Conflict']}</li>"
        f"<li><b style='color:{SCENARIO_COLOR['Severe_Closure']};'>Severe Closure</b> — "
        f"{SCENARIO_NARRATIVES['Severe_Closure']}</li>"
        "</ul>"
        "<h4>Reading the graph</h4>"
        "<p>Teal-filled nodes are the ones for which evidence has "
        "been set (whether by the translator or a manual override). "
        "Unobserved nodes display the most likely state under the "
        "current posterior. Root drivers use distinct color families "
        "so they are easy to distinguish visually.</p>"
        "</div>"
    )


MODEL_OVERVIEW = {
    "labelling": _overview_html(
        "a terminal <b>Scenario</b> node classified from the damage, "
        "duration, and diplomatic-path outcomes"
    ),
    "latent_regime": _overview_html(
        "a latent <b>Scenario</b> regime that <i>generates</i> the damage, "
        "duration, and diplomatic-path outcomes (with context parents US "
        "military response and strait closure)"
    ),
}

PRESENTATION = Presentation(
    scenario_color=SCENARIO_COLOR,
    scenario_label=SCENARIO_LABEL,
    root_driver_style=ROOT_DRIVER_STYLE,
    root_driver_colors=ROOT_DRIVER_COLORS,
    edge_rationales=EDGE_RATIONALES,
    edge_omissions=EDGE_OMISSIONS,
    intermediate_nodes=INTERMEDIATE_NODES,
    model_overview=MODEL_OVERVIEW,
    fallback_keyword_map=FALLBACK_KEYWORD_MAP,
)

__all__ = ["PRESENTATION"]
