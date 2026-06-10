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


# ===== P10 — edge rationale on hover (C11 / V13) ===========================


def test_edge_titles_are_applied_from_the_map():
    edges, levels = TOPOLOGY_LAYOUT["latent_regime"]
    titles = {(s, d): f"why {s}->{d}" for s, d in edges}
    _, out_edges, _ = build_agraph_payload(
        _uniform_marginals(), edges=edges, node_level=levels, edge_titles=titles)
    for e in out_edges:                                  # Edge stores target as `to`
        assert e.title == titles[(e.source, e.to)]


def test_every_edge_carries_its_rationale_title():
    """The gate: with the real edge-rationale map, every rendered edge has a
    non-empty hover title (i.e. the rationale covers every present edge)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
    from components import edge_rationale  # noqa: E402

    for topology in ("labelling", "latent_regime"):
        edges, levels = TOPOLOGY_LAYOUT[topology]
        titles = edge_rationale.edge_title_map(topology)
        _, out_edges, _ = build_agraph_payload(
            _uniform_marginals(), edges=edges, node_level=levels, edge_titles=titles)
        missing = [(e.source, e.to) for e in out_edges if not getattr(e, "title", "")]
        assert not missing, f"{topology}: edges without a rationale title: {missing}"
