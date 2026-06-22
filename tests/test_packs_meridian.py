"""Meridian supply-chain pack (Phase B): structure, CPT coherence, inference."""
from __future__ import annotations

from pgmpy.inference import VariableElimination

from packs.base import Role
from packs.meridian import PACK
from packs.meridian.network import build_network


def test_pack_validates_and_identifies_latent():
    PACK.validate()
    assert PACK.id == "meridian"
    assert PACK.latent == "Disruption_Regime"
    assert PACK.domain == "supply_chain"


def test_node_roles():
    assert PACK.node_meta["Disruption_Regime"].role is Role.LATENT
    assert PACK.node_meta["Supplier_Health"].role is Role.DRIVER
    assert PACK.node_meta["Lead_Time_Slippage"].role is Role.EMISSION
    assert PACK.node_meta["Policy_Headlines"].role is Role.INDICATOR


def test_network_builds_and_is_coherent():
    net = build_network("latent_regime")
    net.check_model()
    assert set(net.nodes()) == set(PACK.states)


def test_presentation_covers_all_latent_states():
    states = set(PACK.states["Disruption_Regime"])
    assert set(PACK.presentation.scenario_color) == states
    assert set(PACK.presentation.scenario_label) == states


def test_emissions_are_monotone_in_severity():
    """The worst emission state must get more mass as the regime worsens — the
    property that lets emissions identify the latent regime."""
    net = build_network("latent_regime")
    cpd = net.get_cpds("Lead_Time_Slippage")
    # P(Blown | regime) should be non-decreasing across Normal→Severe.
    blown_idx = cpd.state_names["Lead_Time_Slippage"].index("Blown")
    regimes = cpd.state_names["Disruption_Regime"]
    vals = [cpd.get_value(Lead_Time_Slippage="Blown", Disruption_Regime=r) for r in regimes]
    assert vals == sorted(vals), f"P(Blown|regime) not monotone: {dict(zip(regimes, vals))}"


def test_evidence_shifts_belief_to_severe():
    net = build_network("latent_regime")
    ve = VariableElimination(net)
    f = ve.query(
        ["Disruption_Regime"],
        evidence={"Lead_Time_Slippage": "Blown", "Force_Majeure_Notices": "Multiple"},
        show_progress=False,
    )
    post = {s: float(f.values[i]) for i, s in enumerate(f.state_names["Disruption_Regime"])}
    assert post["Severe"] > 0.4 and post["Severe"] == max(post.values())
