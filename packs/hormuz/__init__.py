"""Hormuz pack — the original Strait-of-Hormuz crisis demo, as a ScenarioPack.

Self-contained: all Hormuz content lives in this folder (``network`` topology +
CPTs, ``layout`` diagram data, ``headlines`` seeds). The pack imports only leaf
modules (pgmpy/numpy, and the generic ``ExampleHeadline`` dataclass), never an
engine module that reads the ``src.scenario`` seam — so there is no import cycle.
"""
from __future__ import annotations

from pathlib import Path

from packs.base import NodeMeta, Role, ScenarioPack, TranslatorProfile

from packs.hormuz.network import (
    EDGES,
    EDGES_LATENT,
    SCENARIO_NARRATIVES,
    SCENARIO_SIGNATURES,
    STATES,
    build_network,
)
from packs.hormuz.layout import (
    DISPLAY_OVERRIDES,
    NODE_TITLE_WRAP,
    TOPOLOGY_LAYOUT,
)
from packs.hormuz.headlines import EXAMPLE_HEADLINES
from packs.hormuz.presentation import PRESENTATION
from packs.hormuz.seeds import ELICITATION_SEEDS

_LATENT = "Scenario"


def _derive_node_meta(latent, edges_latent, states):
    """Role per node from the latent topology: the latent itself, its direct
    parents (drivers), its direct children (emissions), everything else OTHER.
    Display-only — never touches inference."""
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
            role = Role.OTHER
        meta[node] = NodeMeta(label=DISPLAY_OVERRIDES.get(node, node.replace("_", " ")), role=role)
    return meta


PACK = ScenarioPack(
    id="hormuz",
    title="Strait of Hormuz — crisis-escalation regime",
    domain="geopolitics",
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
    fake_fixtures_dir=str(Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "translator"),
    presentation=PRESENTATION,
    elicitation_seeds=ELICITATION_SEEDS,
    translator_profile=TranslatorProfile(
        domain="the Strait of Hormuz",
        scenario_set_descriptor=(
            "Strait-of-Hormuz scenario set (US–Iran tension, Gulf shipping/energy)"
        ),
        relevance_descriptor=(
            "the Strait of Hormuz / US-Iran tension / Gulf shipping or energy"
        ),
        situation_descriptor="Strait of Hormuz / US-Iran situation",
    ),
    example_headlines=list(EXAMPLE_HEADLINES),
)

__all__ = ["PACK"]
