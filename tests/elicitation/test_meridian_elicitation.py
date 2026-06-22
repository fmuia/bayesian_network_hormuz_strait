"""End-to-end elicitation on the Meridian pack (quadruple-check).

Proves the elicitation engine is pack-agnostic: it runs over Meridian's nodes
with Meridian's calibration seeds and produces a valid, inference-ready Meridian
network with per-CPT kappa — and that the offline role proposal is no longer
Hormuz-flavoured. Imports packs.meridian directly so it is independent of the
suite's hormuz pin (conftest)."""
from __future__ import annotations

import numpy as np

from src.network_spec import NetworkSpec
from src.elicitation.protocols.base import SeedQuestion
from src.elicitation.integration import (
    EffortConfig, ElicitationFramework, ModelSpec, ScriptedClient, run_elicitation,
)
from packs.meridian.network import build_network
from packs.meridian.seeds import ELICITATION_SEEDS

_SEEDS = [SeedQuestion(*r) for r in ELICITATION_SEEDS]
_NODES = ["Disruption_Regime", "Lead_Time_Slippage", "Force_Majeure_Notices"]


def _node_fn(node, config, states):
    rng = np.random.default_rng(abs(hash((node, config))) % (2**32))
    return list(rng.dirichlet(np.ones(len(states)) * 2))


def _factory(spec):
    answers = {s.id: (s.realization * 0.6, s.realization, s.realization * 1.4) for s in _SEEDS}
    return ScriptedClient(spec.model, answers, _node_fn)


def _run():
    base = NetworkSpec.from_pgmpy(build_network("latent_regime"))
    fw = ElicitationFramework(
        name="meridian", models=[ModelSpec("scripted", "calib", "C"), ModelSpec("scripted", "over", "O")],
        n_agents=3, nodes=list(_NODES), seeds=_SEEDS, effort=EffortConfig(n_seeds=6, concurrency=2),
    )
    return base, run_elicitation(base, fw, run_id="m1", created_at="2026-06-22T00:00:00Z", client_factory=_factory)


def test_meridian_elicitation_produces_valid_inference_ready_network():
    base, run = _run()
    run.spec.validate()
    run.spec.to_pgmpy().check_model()
    assert set(run.elicited_nodes) == set(_NODES)
    assert not run.skipped_nodes
    assert all(run.spec.nodes[n].kappa is not None for n in _NODES)  # per-CPT kappa


def test_meridian_seeds_drive_calibration():
    _, run = _run()
    assert {s["id"] for s in run.seeds} == {s.id for s in _SEEDS}
    assert "rare_earth_price_2011" in {s["id"] for s in run.seeds}


def test_offline_roles_are_not_hormuz_flavoured():
    _, run = _run()
    roles = " ".join(r for n in run.elicited_nodes for r in run.nodes[n].roles).lower()
    assert "maritime" not in roles and "energy economist" not in roles
