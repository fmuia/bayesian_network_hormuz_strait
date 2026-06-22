"""Meridian Drive Systems — supply-chain disruption demo, as a ScenarioPack.

Self-contained (network/layout/headlines/presentation under this folder); reached
by the engine through the src.scenario seam, exactly like the Hormuz pack.
"""
from __future__ import annotations

from packs.base import NodeMeta, Role, ScenarioPack, TranslatorProfile

from packs.meridian.network import (
    EDGES,
    EDGES_LATENT,
    SCENARIO_NARRATIVES,
    SCENARIO_SIGNATURES,
    STATES,
    build_network,
)
from packs.meridian.layout import DISPLAY_OVERRIDES, NODE_TITLE_WRAP, TOPOLOGY_LAYOUT
from packs.meridian.headlines import EXAMPLE_HEADLINES
from packs.meridian.presentation import PRESENTATION
from packs.meridian.seeds import ELICITATION_SEEDS

_LATENT = "Disruption_Regime"


def _derive_node_meta(latent, edges_latent, states):
    parents = {a for a, b in edges_latent if b == latent}
    children = {b for a, b in edges_latent if a == latent}
    meta = {}
    for node in states:
        if node == latent:
            role = Role.LATENT
        elif node in parents:
            role = Role.DRIVER
        elif node in children:
            role = Role.EMISSION
        else:
            role = Role.INDICATOR if node == "Policy_Headlines" else Role.OTHER
        meta[node] = NodeMeta(label=DISPLAY_OVERRIDES.get(node, node.replace("_", " ")), role=role)
    return meta


PACK = ScenarioPack(
    id="meridian",
    title="Meridian Drive Systems — supply-chain disruption",
    domain="supply_chain",
    states=STATES,
    build_network=build_network,
    latent=_LATENT,
    edges=EDGES,
    edges_latent=EDGES_LATENT,
    narratives=SCENARIO_NARRATIVES,
    signatures=SCENARIO_SIGNATURES,
    node_meta=_derive_node_meta(_LATENT, EDGES_LATENT, STATES),
    layout=TOPOLOGY_LAYOUT,
    display_overrides=DISPLAY_OVERRIDES,
    node_title_wrap=NODE_TITLE_WRAP,
    presentation=PRESENTATION,
    elicitation_seeds=ELICITATION_SEEDS,
    translator_profile=TranslatorProfile(
        domain="Meridian's EV-motor supply chain",
        scenario_set_descriptor=(
            "Meridian supply-chain disruption set (rare-earth magnets, power "
            "semiconductors, inbound logistics)"
        ),
        relevance_descriptor=(
            "Meridian's supply chain / rare-earth magnets / power semiconductors / "
            "inbound logistics / supplier financial health"
        ),
        situation_descriptor="Meridian supply-chain disruption situation",
    ),
    example_headlines=list(EXAMPLE_HEADLINES),
)

__all__ = ["PACK"]
