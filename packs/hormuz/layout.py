"""Hormuz diagram layout + display data (moved out of src/viz.py).

Pack-specific presentation: per-node DAG levels for each topology, the
topology→(edges, levels) map the renderer consumes, and the display-name
overrides/wraps. The renderer in ``src.viz`` is generic and reads these via the
``src.scenario`` seam.
"""
from __future__ import annotations

from typing import Dict

from packs.hormuz.network import EDGES, EDGES_LATENT

_NODE_LEVEL: Dict[str, int] = {
    "US_Iran_Negotiations": 0,
    "Iranian_Regime_Stability": 0,
    "Third_Party_Mediation": 0,
    "Sanctions_Trajectory": 0,
    "Iran_Aligned_Militia_Attacks": 1,
    "Tanker_Incidents": 1,
    "US_Military_Response": 2,
    "Strait_Operationally_Closed": 2,
    "Energy_Infrastructure_Damage": 3,
    "Conflict_Duration": 3,
    "Diplomatic_Resolution_Path": 3,
    "Oil_Price_Regime": 3,
    "Scenario": 4,
}

# Latent-regime layout: Scenario sits between its parents {M, C} (level 2) and its
# emissions {D, T, P} (level 4); Oil_Price is a child of D so it drops to level 5.
_NODE_LEVEL_LATENT: Dict[str, int] = {
    **_NODE_LEVEL,
    "Scenario": 3,
    "Energy_Infrastructure_Damage": 4,
    "Conflict_Duration": 4,
    "Diplomatic_Resolution_Path": 4,
    "Oil_Price_Regime": 5,
}

# Map a topology name to (edges, node-level) so callers can pass a single string.
TOPOLOGY_LAYOUT = {
    "labelling": (EDGES, _NODE_LEVEL),
    "latent_regime": (EDGES_LATENT, _NODE_LEVEL_LATENT),
}

# Display-name overrides (default transform is underscore→space); only the
# hyphenated outlet needs special-casing.
DISPLAY_OVERRIDES: Dict[str, str] = {
    "Iran_Aligned_Militia_Attacks": "Iran-Aligned Militia Attacks",
}

# Two-line wrapping for long node titles in the rendered diagram.
NODE_TITLE_WRAP: Dict[str, str] = {
    "US Iran Negotiations": "US Iran\nNegotiations",
    "Iranian Regime Stability": "Iranian Regime\nStability",
    "Third Party Mediation": "Third Party\nMediation",
    "Sanctions Trajectory": "Sanctions\nTrajectory",
    "Iran-Aligned Militia Attacks": "Iran-Aligned Militia\nAttacks",
    "Tanker Incidents": "Tanker\nIncidents",
    "US Military Response": "US Military\nResponse",
    "Strait Operationally Closed": "Strait Operationally\nClosed",
    "Energy Infrastructure Damage": "Energy Infrastructure\nDamage",
    "Conflict Duration": "Conflict\nDuration",
    "Diplomatic Resolution Path": "Diplomatic Resolution\nPath",
    "Oil Price Regime": "Oil Price\nRegime",
}

__all__ = [
    "TOPOLOGY_LAYOUT", "DISPLAY_OVERRIDES", "NODE_TITLE_WRAP",
    "_NODE_LEVEL", "_NODE_LEVEL_LATENT",
]
