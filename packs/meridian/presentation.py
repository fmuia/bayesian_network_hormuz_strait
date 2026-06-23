"""Meridian presentation data (colours, labels, edge rationales, model copy,
fallback-translator keyword map)."""
from __future__ import annotations

from packs.base import Presentation
from packs.meridian.network import SCENARIO_NARRATIVES

# --- latent-state colours / labels (Normal → Severe gradient) ---------------
SCENARIO_COLOR = {
    "Normal": "#2E8B57",
    "Single_Supplier_Stress": "#D4A017",
    "Multi_Node_Ripple": "#E8743B",
    "Severe": "#B22222",
}
SCENARIO_LABEL = {
    "Normal": "Normal",
    "Single_Supplier_Stress": "Single-supplier stress",
    "Multi_Node_Ripple": "Multi-node ripple",
    "Severe": "Severe",
}

# --- root-driver styling ----------------------------------------------------
ROOT_DRIVER_STYLE = {
    "Geo_Exposure": ("#DBEAFE", "#1D4ED8"),
    "Supplier_Health": ("#FCE7F3", "#BE185D"),
    "Route_Status": ("#FEF3C7", "#B45309"),
}
ROOT_DRIVER_COLORS = {
    "Geo_Exposure": ("#DBEAFE", "#1D4ED8", "#1E40AF"),
    "Supplier_Health": ("#FCE7F3", "#BE185D", "#9D174D"),
    "Route_Status": ("#FEF3C7", "#B45309", "#92400E"),
}

# --- audit-view intermediate-node ordering (non-root, non-latent) ----------
# mechanisms → indicator → emissions → impacts, top-to-bottom of the DAG.
INTERMEDIATE_NODES = [
    "Magnet_Supply", "Power_Semi_Supply",
    "Policy_Headlines",
    "Lead_Time_Slippage", "Force_Majeure_Notices", "Input_Price_Spike", "Expedite_Spend",
    "Production_Output", "On_Time_Delivery", "Gross_Margin_Hit",
]

# --- deterministic offline (fake) translator keyword map -------------------
FALLBACK_KEYWORD_MAP = [
    (("export control", "licens", "tariff", "sanction", "ban"), "Geo_Exposure", "High"),
    (("bond", "default", "insolven", "downgrade", "distress"), "Supplier_Health", "Distressed"),
    (("port", "congestion", "blank sailing", "blocked", "closure"), "Route_Status", "Disrupted"),
    (("lead time", "lead-time", "delay", "backlog"), "Lead_Time_Slippage", "Blown"),
    (("force majeure", "shutdown", "halt"), "Force_Majeure_Notices", "Multiple"),
    (("price", "spot", "spike", "surge", "%"), "Input_Price_Spike", "Spiking"),
    (("expedite", "air freight", "air-freight", "premium freight"), "Expedite_Spend", "Surging"),
    (("normal", "stable", "ease", "normalise", "normalize"), "Lead_Time_Slippage", "OnTime"),
]

# --- edge rationale / omission tables (single topology) --------------------
_EDGE_RATIONALE = [
    # drivers → mechanisms (the two critical-input bottlenecks)
    ("Geo_Exposure", "Magnet_Supply",
     "Export controls, tariffs and sanctions on rare-earth elements are the "
     "dominant driver of magnet availability — weighted highest in the magnet "
     "mechanism CPT."),
    ("Supplier_Health", "Magnet_Supply",
     "A financially distressed magnet supplier tightens magnet availability "
     "independently of geopolitics."),
    ("Route_Status", "Power_Semi_Supply",
     "Congested or blocked inbound lanes are the dominant driver of power-"
     "semiconductor availability (long Asia/Europe fab-to-plant routes)."),
    ("Supplier_Health", "Power_Semi_Supply",
     "A distressed power-semi supplier tightens availability independently of "
     "logistics — so Supplier_Health couples both bottlenecks."),
    # mechanisms → regime
    ("Magnet_Supply", "Disruption_Regime",
     "Critical magnet supply is the single largest contributor to a disruption "
     "regime for an EV-motor maker."),
    ("Power_Semi_Supply", "Disruption_Regime",
     "Critical power-semiconductor supply pushes the regime toward multi-node "
     "and severe states."),
    # regime → emissions
    ("Disruption_Regime", "Lead_Time_Slippage",
     "Emission: the regime drives delivery performance — slippage appears early, "
     "blown lead times concentrate in the Severe regime."),
    ("Disruption_Regime", "Force_Majeure_Notices",
     "Emission: clustered force-majeure declarations are characteristic of the "
     "Multi-node and Severe regimes."),
    ("Disruption_Regime", "Input_Price_Spike",
     "Emission: broad input-price spikes accompany multi-node and severe "
     "disruption."),
    ("Geo_Exposure", "Input_Price_Spike",
     "Explaining-away: input prices also move with macro rare-earth pricing, so a "
     "price spike under high geopolitical exposure is partly explained by geo and "
     "implicates the regime less."),
    ("Disruption_Regime", "Expedite_Spend",
     "Emission: premium/air-freight spend to recover schedule surges late, in the "
     "Multi-node and Severe regimes."),
    # regime → impacts → downstream impacts (the P&L story)
    ("Disruption_Regime", "Production_Output",
     "Impact: a disrupted regime starves the line of magnets/power-semis, cutting "
     "buildable units — Halted concentrates in the Severe regime."),
    ("Production_Output", "On_Time_Delivery",
     "Impact: lost production flows straight to missed OEM deliveries (and the "
     "contractual penalties that follow)."),
    ("Disruption_Regime", "Gross_Margin_Hit",
     "Impact: the regime hits margin directly via expedite premiums and input-"
     "price spikes."),
    ("Production_Output", "Gross_Margin_Hit",
     "Impact: lost volume compounds the margin hit through fixed-cost under-"
     "absorption."),
    # driver indicator
    ("Geo_Exposure", "Policy_Headlines",
     "Indicator: higher geopolitical exposure generates more escalating policy "
     "headlines — the cause-side signal the translator reads."),
]
_EDGE_OMISSIONS = [
    ("Supplier_Health", "Disruption_Regime",
     "Mediated by the two mechanism nodes: supplier health reaches the regime "
     "through magnet and power-semi availability, not directly."),
    ("Disruption_Regime", "On_Time_Delivery",
     "Mediated by Production_Output: the regime affects deliveries only through "
     "the volume it removes from the line."),
]

EDGE_RATIONALES = {"labelling": _EDGE_RATIONALE, "latent_regime": _EDGE_RATIONALE}
EDGE_OMISSIONS = {"labelling": _EDGE_OMISSIONS, "latent_regime": _EDGE_OMISSIONS}

# --- model-overview copy ----------------------------------------------------
def _overview_html() -> str:
    return (
        "<div class='explain'>"
        "<p>The Bayesian network tracks a latent <b>Disruption Regime</b> for "
        "Meridian's EV-motor supply chain. It is a four-layer DAG, not a flat "
        "classifier — that structure is the point.</p>"
        "<h4>Four layers</h4>"
        "<p><b>Drivers</b> (geopolitical exposure, supplier financial health, "
        "logistics-route status) feed two <b>mechanism</b> nodes — the critical-"
        "input bottlenecks <code>Magnet_Supply</code> and "
        "<code>Power_Semi_Supply</code> — which in turn set the <b>regime</b>. The "
        "regime generates observable <b>emissions</b> (the signals an analyst "
        "feeds in) and propagates to a downstream <b>impact</b> layer "
        "(<code>Production_Output → On_Time_Delivery</code>, "
        "<code>Gross_Margin_Hit</code>) — so a headline updates a P&amp;L-relevant "
        "node, not just an abstract regime.</p>"
        "<h4>What the structure buys</h4>"
        "<ul>"
        "<li><b>Mechanism</b> — drivers reach the regime through two named "
        "bottlenecks; <code>Supplier_Health</code> couples both.</li>"
        "<li><b>Explaining-away</b> — <code>Input_Price_Spike</code> has both the "
        "regime and <code>Geo_Exposure</code> as parents, so a spike under high "
        "geopolitical exposure is partly explained by macro rare-earth pricing and "
        "implicates the regime less.</li>"
        "<li><b>Diagnostic + predictive flow</b> — feed signals to infer the "
        "regime, and read the impact layer to price the consequences.</li>"
        "</ul>"
        "<h4>Translator</h4>"
        "<p>A free-text headline is passed through an LLM translator that extracts "
        "BN-relevant assignments (e.g. <i>\"magnet lead times stretch to 14 weeks\"</i> "
        "gives high probability to <code>Lead_Time_Slippage = Blown</code>). Those "
        "soft assignments become evidence; variable elimination yields the posterior "
        "over the regime and every node.</p>"
        "<h4>Regime definitions</h4>"
        "<ul>"
        f"<li><b style='color:{SCENARIO_COLOR['Normal']};'>Normal</b> — {SCENARIO_NARRATIVES['Normal']}</li>"
        f"<li><b style='color:{SCENARIO_COLOR['Single_Supplier_Stress']};'>Single-supplier stress</b> — "
        f"{SCENARIO_NARRATIVES['Single_Supplier_Stress']}</li>"
        f"<li><b style='color:{SCENARIO_COLOR['Multi_Node_Ripple']};'>Multi-node ripple</b> — "
        f"{SCENARIO_NARRATIVES['Multi_Node_Ripple']}</li>"
        f"<li><b style='color:{SCENARIO_COLOR['Severe']};'>Severe</b> — {SCENARIO_NARRATIVES['Severe']}</li>"
        "</ul>"
        "<h4>Reading the graph</h4>"
        "<p>Teal-filled nodes have evidence set. Unobserved nodes show the most "
        "likely state under the current posterior. Drivers use distinct colour "
        "families so they are easy to distinguish.</p>"
        "</div>"
    )


MODEL_OVERVIEW = {"labelling": _overview_html(), "latent_regime": _overview_html()}

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
