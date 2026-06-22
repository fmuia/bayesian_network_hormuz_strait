"""The src.scenario seam (Phase A.3): with the default pack (hormuz), every
symbol the seam exposes is the *same object* as the legacy src.network export,
so routing consumers through the seam is behaviour-preserving."""
from __future__ import annotations

import packs.hormuz.network as N
import src.scenario as S


def test_seam_reexports_identical_objects():
    assert S.STATES is N.STATES
    assert S.EDGES is N.EDGES
    assert S.EDGES_LATENT is N.EDGES_LATENT
    assert S.SCENARIO_NARRATIVES is N.SCENARIO_NARRATIVES
    assert S.SCENARIO_SIGNATURES is N.SCENARIO_SIGNATURES
    assert S.build_network is N.build_network


def test_seam_active_pack_is_hormuz_by_default():
    assert S.PACK.id == "hormuz"
    assert S.LATENT == "Scenario"


def test_seam_build_network_constructs():
    net = S.build_network("latent_regime")
    assert set(net.nodes()) == set(S.STATES)
    net.check_model()
