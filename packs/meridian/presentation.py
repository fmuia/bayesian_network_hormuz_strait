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
INTERMEDIATE_NODES = [
    "Policy_Headlines", "Lead_Time_Slippage", "Force_Majeure_Notices",
    "Input_Price_Spike", "Expedite_Spend",
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
    ("Geo_Exposure", "Disruption_Regime",
     "Export controls, tariffs and sanctions on critical inputs (e.g. rare-earth "
     "magnets) raise the prior on a disruption regime before any operational signal."),
    ("Supplier_Health", "Disruption_Regime",
     "A financially distressed critical supplier is the single largest driver of "
     "a supplier-side disruption — weighted highest in the latent CPT."),
    ("Route_Status", "Disruption_Regime",
     "Congested or blocked inbound lanes/ports transmit disruption to the network "
     "even when suppliers are healthy."),
    ("Disruption_Regime", "Lead_Time_Slippage",
     "Emission: the regime drives delivery performance — slippage appears early, "
     "blown lead times concentrate in the Severe regime."),
    ("Disruption_Regime", "Force_Majeure_Notices",
     "Emission: clustered force-majeure declarations are characteristic of the "
     "Multi-node and Severe regimes."),
    ("Disruption_Regime", "Input_Price_Spike",
     "Emission: broad input-price spikes lag but accompany multi-node and severe "
     "disruption."),
    ("Disruption_Regime", "Expedite_Spend",
     "Emission: premium/air-freight spend to recover schedule surges late, in the "
     "Multi-node and Severe regimes."),
    ("Geo_Exposure", "Policy_Headlines",
     "Indicator: higher geopolitical exposure generates more escalating policy "
     "headlines — the cause-side signal the translator reads."),
]
_EDGE_OMISSIONS = [
    ("Route_Status", "Input_Price_Spike",
     "Mediated by the disruption regime: route status reaches prices through the "
     "overall regime, not directly."),
]

EDGE_RATIONALES = {"labelling": _EDGE_RATIONALE, "latent_regime": _EDGE_RATIONALE}
EDGE_OMISSIONS = {"labelling": _EDGE_OMISSIONS, "latent_regime": _EDGE_OMISSIONS}

# --- model-overview copy ----------------------------------------------------
def _overview_html() -> str:
    return (
        "<div class='explain'>"
        "<p>The Bayesian network tracks a latent <b>Disruption Regime</b> for "
        "Meridian's EV-motor supply chain, inferred from observable signals given "
        "three <b>drivers</b> (geopolitical exposure, supplier financial health, "
        "logistics-route status).</p>"
        "<h4>Two layers</h4>"
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
