"""Tests for Layer 5: NetworkSpec, per-CPT kappa, and store round-trips."""

from __future__ import annotations

import numpy as np
from pgmpy.inference import VariableElimination

from src.elicitation.db import create_all, make_engine, make_session_factory
from src.elicitation.export import (
    cpts_to_network_spec,
    network_spec_to_cpts,
    spec_from_dict,
    spec_to_dict,
)
from src.scenario import build_network
from src.network_spec import NetworkSpec
from src.sensitivity import node_credible_intervals


def _posterior(net, node: str, evidence: dict) -> np.ndarray:
    f = VariableElimination(net).query([node], evidence=evidence, show_progress=False)
    return np.asarray(f.values, dtype=float)


def test_pgmpy_roundtrip_preserves_posteriors() -> None:
    base = build_network()
    spec = NetworkSpec.from_pgmpy(base)
    rebuilt = spec.to_pgmpy()
    evidence = {"Iran_Aligned_Militia_Attacks": "high", "US_Iran_Negotiations": "breakdown"}
    np.testing.assert_allclose(
        _posterior(base, "Scenario", evidence),
        _posterior(rebuilt, "Scenario", evidence),
        atol=1e-9,
    )


def test_spec_dict_roundtrip() -> None:
    spec = NetworkSpec.from_pgmpy(build_network())
    rebuilt = spec_from_dict(spec_to_dict(spec))
    assert set(rebuilt.nodes) == set(spec.nodes)
    a = spec.nodes["Tanker_Incidents"]
    b = rebuilt.nodes["Tanker_Incidents"]
    assert a.parents == b.parents and a.states == b.states
    for key in a.cpt:
        np.testing.assert_allclose(a.cpt[key], b.cpt[key])


def test_kappa_map_collects_per_cpt_kappa() -> None:
    spec = NetworkSpec.from_pgmpy(build_network())
    spec.nodes["Tanker_Incidents"].kappa = 5.0
    spec.nodes["Scenario"].kappa = 40.0
    km = spec.kappa_map()
    assert km == {"Tanker_Incidents": 5.0, "Scenario": 40.0}


def test_per_cpt_kappa_widens_uncertain_node() -> None:
    """A CPT marked 'uncertain' (low kappa) yields a wider credible interval
    than one marked 'tight' (high kappa), via the patched resampler."""
    base = build_network()
    evidence = {"Iran_Aligned_Militia_Attacks": "high"}

    def width(kappa: float) -> float:
        # per-CPT concentration map: low/high kappa on Tanker_Incidents, 20 elsewhere
        concentration = {v: (kappa if v == "Tanker_Incidents" else 20.0) for v in base.nodes()}
        ci = node_credible_intervals(
            evidence,
            nodes=["Tanker_Incidents"],
            m=120,
            seed=0,
            base_network=base,
            concentration=concentration,
        )["Tanker_Incidents"]
        # total interval width summed across states
        return sum(hi - lo for (_, lo, hi) in ci.values())

    assert width(5.0) > width(200.0)


def test_db_roundtrip_preserves_cpts_and_kappa() -> None:
    spec = NetworkSpec.from_pgmpy(build_network())
    spec.nodes["Tanker_Incidents"].kappa = 12.0
    spec.nodes["Tanker_Incidents"].kappa_level = "normal"

    engine = make_engine("sqlite:///:memory:")
    create_all(engine)
    Session = make_session_factory(engine)
    with Session() as s:
        network_id = network_spec_to_cpts(s, spec, "hormuz", topology="scenario_child")
    with Session() as s:
        restored = cpts_to_network_spec(s, network_id)

    assert set(restored.nodes) == set(spec.nodes)
    t = restored.nodes["Tanker_Incidents"]
    assert t.kappa == 12.0 and t.kappa_level == "normal"
    assert t.parents == spec.nodes["Tanker_Incidents"].parents
    for key, col in spec.nodes["Tanker_Incidents"].cpt.items():
        np.testing.assert_allclose(t.cpt[key], col)
    # restored spec rebuilds into a valid, inference-ready network
    restored.to_pgmpy().check_model()
