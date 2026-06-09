"""P3 (Plan 5 E2) — tests for src/viz.py build_agraph_payload, both topologies."""
from __future__ import annotations

import pytest

from src.network import STATES
from src.viz import (
    _NAVY,
    _ROOT_DRIVER_COLORS,
    _TEAL,
    TOPOLOGY_LAYOUT,
    build_agraph_payload,
)


def _uniform_marginals():
    return {node: {s: 1.0 / len(states) for s in states} for node, states in STATES.items()}


@pytest.mark.parametrize("topology", ["labelling", "latent_regime"])
def test_payload_counts_and_levels(topology):
    edges, levels = TOPOLOGY_LAYOUT[topology]
    nodes, out_edges, _ = build_agraph_payload(
        _uniform_marginals(), edges=edges, node_level=levels)
    assert {n.id for n in nodes} == set(STATES)        # every node rendered
    assert len(out_edges) == len(list(edges))          # every edge rendered
    for n in nodes:
        assert n.level == levels.get(n.id, 0)          # topological level honoured
    by_id = {n.id: n.level for n in nodes}
    assert by_id["Scenario"] == levels["Scenario"]     # differs by topology (4 vs 3)


def test_observed_non_root_node_uses_observed_fill():
    edges, levels = TOPOLOGY_LAYOUT["latent_regime"]
    nodes, _, _ = build_agraph_payload(
        _uniform_marginals(), observed={"Tanker_Incidents": "frequent"},
        edges=edges, node_level=levels)
    by_id = {n.id: n for n in nodes}
    assert by_id["Tanker_Incidents"].color["background"] == _TEAL


def test_root_driver_uses_dedicated_colour_family():
    edges, levels = TOPOLOGY_LAYOUT["latent_regime"]
    nodes, _, _ = build_agraph_payload(_uniform_marginals(), edges=edges, node_level=levels)
    by_id = {n.id: n for n in nodes}
    root = next(iter(_ROOT_DRIVER_COLORS))
    assert by_id[root].color["background"] == _ROOT_DRIVER_COLORS[root][0]


def test_scenario_node_is_navy():
    nodes, _, _ = build_agraph_payload(_uniform_marginals())
    by_id = {n.id: n for n in nodes}
    assert by_id["Scenario"].color["background"] == _NAVY
