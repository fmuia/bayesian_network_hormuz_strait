"""Hormuz pack adapter (Phase A.2): the pack exposes the existing scenario
content unchanged and builds the same network."""
from __future__ import annotations

from packs.base import Role
from packs.hormuz import PACK
from packs import registry
import packs.hormuz.network as N


def test_pack_validates_and_identifies_latent():
    PACK.validate()
    assert PACK.id == "hormuz"
    assert PACK.latent == "Scenario"


def test_pack_content_is_the_engine_content():
    assert PACK.states is N.STATES
    assert PACK.edges_latent is N.EDGES_LATENT
    assert PACK.narratives is N.SCENARIO_NARRATIVES


def test_node_meta_roles_match_latent_topology():
    assert PACK.node_meta["Scenario"].role is Role.LATENT
    # children of Scenario in the latent topology are emissions …
    assert PACK.node_meta["Energy_Infrastructure_Damage"].role is Role.EMISSION
    # … and Pa(S) are drivers
    assert PACK.node_meta["US_Military_Response"].role is Role.DRIVER


def test_build_network_through_pack_matches_engine():
    net = PACK.build_network("latent_regime")
    assert set(net.nodes()) == set(N.STATES)
    net.check_model()   # CPDs consistent


def test_registry_resolves_hormuz():
    registry._reset_cache()
    assert registry.get_pack("hormuz") is PACK
