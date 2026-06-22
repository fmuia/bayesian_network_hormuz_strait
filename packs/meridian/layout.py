"""Meridian diagram layout + display data."""
from __future__ import annotations

from typing import Dict

from packs.meridian.network import EDGES_LATENT

_NODE_LEVEL: Dict[str, int] = {
    "Geo_Exposure": 0,
    "Supplier_Health": 0,
    "Route_Status": 0,
    "Policy_Headlines": 1,
    "Disruption_Regime": 1,
    "Lead_Time_Slippage": 2,
    "Force_Majeure_Notices": 2,
    "Input_Price_Spike": 2,
    "Expedite_Spend": 2,
}

# Single topology — expose both keys so engine code referencing either works.
TOPOLOGY_LAYOUT = {
    "labelling": (EDGES_LATENT, _NODE_LEVEL),
    "latent_regime": (EDGES_LATENT, _NODE_LEVEL),
}

DISPLAY_OVERRIDES: Dict[str, str] = {}

NODE_TITLE_WRAP: Dict[str, str] = {
    "Disruption Regime": "Disruption\nRegime",
    "Supplier Health": "Supplier\nHealth",
    "Route Status": "Route\nStatus",
    "Policy Headlines": "Policy\nHeadlines",
    "Lead Time Slippage": "Lead-Time\nSlippage",
    "Force Majeure Notices": "Force-Majeure\nNotices",
    "Input Price Spike": "Input-Price\nSpike",
    "Expedite Spend": "Expedite\nSpend",
}

__all__ = ["TOPOLOGY_LAYOUT", "DISPLAY_OVERRIDES", "NODE_TITLE_WRAP"]
