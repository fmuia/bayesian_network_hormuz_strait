"""The scenario seam: the single place the engine reads scenario *content*.

Engine and app modules import scenario constants from here
(``from src.scenario import STATES``) instead of from a specific scenario module.
The active pack is resolved once, from ``SCENARIO_PACK`` (default ``hormuz``), so
flipping the env var swaps the whole demo without touching any consumer.

Each pack is self-contained (its content lives under ``packs/<id>/``); the seam
re-exports the active pack's objects, proven byte-identical for Hormuz by
``tests/test_scenario_seam.py`` and the golden snapshot. Use :func:`reload` in
tests that switch packs mid-process.
"""
from __future__ import annotations

from packs.registry import get_active

_pack = get_active()

# --- scenario content (identical objects to the active pack) ----------------
PACK = _pack
STATES = _pack.states
EDGES = _pack.edges
EDGES_LATENT = _pack.edges_latent
SCENARIO_NARRATIVES = _pack.narratives
SCENARIO_SIGNATURES = _pack.signatures
NODE_META = _pack.node_meta
LAYOUT = _pack.layout
DISPLAY_OVERRIDES = _pack.display_overrides
NODE_TITLE_WRAP = _pack.node_title_wrap
PRESENTATION = _pack.presentation
TRANSLATOR_PROFILE = _pack.translator_profile
FAKE_FIXTURES_DIR = _pack.fake_fixtures_dir
ELICITATION_SEEDS = _pack.elicitation_seeds
EXAMPLE_HEADLINES = _pack.example_headlines
LATENT = _pack.latent
build_network = _pack.build_network


def reload():
    """Re-resolve the active pack (after changing ``SCENARIO_PACK``). Returns the
    new pack. Rebinds this module's globals so subsequent attribute reads see it.
    Module-level ``from src.scenario import STATES`` bindings in already-imported
    consumers are NOT updated — use in tests that import lazily."""
    global _pack, PACK, STATES, EDGES, EDGES_LATENT, SCENARIO_NARRATIVES
    global SCENARIO_SIGNATURES, NODE_META, LAYOUT, DISPLAY_OVERRIDES, NODE_TITLE_WRAP
    global PRESENTATION, TRANSLATOR_PROFILE, FAKE_FIXTURES_DIR, ELICITATION_SEEDS
    global EXAMPLE_HEADLINES, LATENT, build_network
    from packs import registry

    registry._reset_cache()
    _pack = get_active()
    PACK = _pack
    STATES = _pack.states
    EDGES = _pack.edges
    EDGES_LATENT = _pack.edges_latent
    SCENARIO_NARRATIVES = _pack.narratives
    SCENARIO_SIGNATURES = _pack.signatures
    NODE_META = _pack.node_meta
    LAYOUT = _pack.layout
    DISPLAY_OVERRIDES = _pack.display_overrides
    NODE_TITLE_WRAP = _pack.node_title_wrap
    PRESENTATION = _pack.presentation
    TRANSLATOR_PROFILE = _pack.translator_profile
    FAKE_FIXTURES_DIR = _pack.fake_fixtures_dir
    ELICITATION_SEEDS = _pack.elicitation_seeds
    EXAMPLE_HEADLINES = _pack.example_headlines
    LATENT = _pack.latent
    build_network = _pack.build_network
    return _pack


__all__ = [
    "PACK", "STATES", "EDGES", "EDGES_LATENT", "SCENARIO_NARRATIVES",
    "SCENARIO_SIGNATURES", "NODE_META", "LAYOUT", "DISPLAY_OVERRIDES",
    "NODE_TITLE_WRAP", "PRESENTATION", "TRANSLATOR_PROFILE", "FAKE_FIXTURES_DIR",
    "ELICITATION_SEEDS", "EXAMPLE_HEADLINES", "LATENT", "build_network", "reload",
]
