"""Pack scaffolding: the ScenarioPack contract and its validation (Phase A.1).

Pure and isolated — does not import any pack folder or the engine, so it pins
the contract independently of the scenario content extracted in later phases.
"""
from __future__ import annotations

import pytest

from packs.base import NodeMeta, Role, ScenarioPack
from packs import registry


def _toy(**over) -> ScenarioPack:
    base = dict(
        id="toy", title="Toy", domain="test",
        states={"A": ["lo", "hi"], "S": ["x", "y"]},
        build_network=lambda topology="latent_regime": None,
        latent="S",
        edges_latent=[("A", "S")],
        narratives={"x": "…", "y": "…"},
        node_meta={"S": NodeMeta(label="S", role=Role.LATENT)},
        layout={"A": (0, 0), "S": (1, 1)},
    )
    base.update(over)
    return ScenarioPack(**base)


def test_valid_pack_passes():
    _toy().validate()   # no raise


def test_empty_states_rejected():
    with pytest.raises(ValueError, match="empty state"):
        _toy(states={}).validate()


def test_latent_must_be_a_known_node():
    with pytest.raises(ValueError, match="latent"):
        _toy(latent="ghost").validate()


def test_edge_to_unknown_node_rejected():
    with pytest.raises(ValueError, match="unknown node"):
        _toy(edges_latent=[("A", "ghost")]).validate()


def test_narrative_for_non_latent_state_rejected():
    with pytest.raises(ValueError, match="non-latent-states"):
        _toy(narratives={"x": "…", "z": "stray"}).validate()


def test_node_meta_with_unknown_node_rejected():
    with pytest.raises(ValueError, match="unknown node"):
        _toy(node_meta={"ghost": NodeMeta(label="ghost")}).validate()


def test_opaque_layout_is_not_key_checked():
    # layout may be topology-keyed (as Hormuz's is), not node-keyed — no raise.
    _toy(layout={"latent_regime": (["edges"], {})}).validate()


def test_registry_unknown_pack_raises():
    registry._reset_cache()
    with pytest.raises(KeyError, match="Unknown scenario pack"):
        registry.get_pack("does_not_exist")


def test_registry_default_and_active_id(monkeypatch):
    monkeypatch.delenv("SCENARIO_PACK", raising=False)
    assert registry.active_pack_id() == registry.DEFAULT_PACK
    monkeypatch.setenv("SCENARIO_PACK", "meridian")
    assert registry.active_pack_id() == "meridian"
