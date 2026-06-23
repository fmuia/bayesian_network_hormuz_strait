"""Scenario packs: self-contained content bundles (topology, CPTs, layout,
narratives, example headlines, optional decision layer) that the shared engine
consumes. One pack per scenario — Hormuz, Meridian, Taiwan, …

The engine is scenario-blind; everything scenario-specific lives in a pack and
is reached through :mod:`packs.registry` (and the ``src.scenario`` seam).
"""
