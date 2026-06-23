"""Meridian diagram layout + display data (four-layer DAG)."""
from __future__ import annotations

from typing import Dict

from packs.meridian.network import EDGES_LATENT

# Hierarchical levels: drivers → mechanisms/indicator → regime → emissions &
# production → downstream impacts.
_NODE_LEVEL: Dict[str, int] = {
    "Geo_Exposure": 0,
    "Supplier_Health": 0,
    "Route_Status": 0,
    "Magnet_Supply": 1,
    "Power_Semi_Supply": 1,
    "Policy_Headlines": 1,
    "Disruption_Regime": 2,
    "Lead_Time_Slippage": 3,
    "Force_Majeure_Notices": 3,
    "Input_Price_Spike": 3,
    "Expedite_Spend": 3,
    "Production_Output": 3,
    "On_Time_Delivery": 4,
    "Gross_Margin_Hit": 4,
}

# Single topology — expose both keys so engine code referencing either works.
TOPOLOGY_LAYOUT = {
    "labelling": (EDGES_LATENT, _NODE_LEVEL),
    "latent_regime": (EDGES_LATENT, _NODE_LEVEL),
}

DISPLAY_OVERRIDES: Dict[str, str] = {
    "Power_Semi_Supply": "Power-Semi Supply",
    "On_Time_Delivery": "On-Time Delivery",
}

NODE_TITLE_WRAP: Dict[str, str] = {
    "Geo Exposure": "Geo\nExposure",
    "Supplier Health": "Supplier\nHealth",
    "Route Status": "Route\nStatus",
    "Magnet Supply": "Magnet\nSupply",
    "Power-Semi Supply": "Power-Semi\nSupply",
    "Policy Headlines": "Policy\nHeadlines",
    "Disruption Regime": "Disruption\nRegime",
    "Lead Time Slippage": "Lead-Time\nSlippage",
    "Force Majeure Notices": "Force-Majeure\nNotices",
    "Input Price Spike": "Input-Price\nSpike",
    "Expedite Spend": "Expedite\nSpend",
    "Production Output": "Production\nOutput",
    "On-Time Delivery": "On-Time\nDelivery",
    "Gross Margin Hit": "Gross-Margin\nHit",
}

__all__ = ["TOPOLOGY_LAYOUT", "DISPLAY_OVERRIDES", "NODE_TITLE_WRAP"]
