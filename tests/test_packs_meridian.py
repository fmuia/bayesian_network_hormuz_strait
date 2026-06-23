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


# --- the four-layer structure (would all stay green on a naive-Bayes star) ---

def test_drivers_are_mediated_by_mechanism_layer():
    """No driver wires straight into the regime: each reaches it through a
    mechanism node (Magnet_Supply / Power_Semi_Supply)."""
    net = build_network("latent_regime")
    regime_parents = set(net.get_parents("Disruption_Regime"))
    assert regime_parents == {"Magnet_Supply", "Power_Semi_Supply"}
    for driver in ("Geo_Exposure", "Supplier_Health", "Route_Status"):
        assert ("Disruption_Regime") not in net.successors(driver)
    # Supplier_Health is the shared parent that couples both bottlenecks.
    assert {"Magnet_Supply", "Power_Semi_Supply"}.issubset(set(net.successors("Supplier_Health")))


def test_mechanism_and_impact_nodes_are_intermediate():
    for n in ("Magnet_Supply", "Power_Semi_Supply", "Production_Output",
              "On_Time_Delivery", "Gross_Margin_Hit"):
        assert PACK.node_meta[n].role is Role.OTHER


def test_input_price_spike_has_explaining_away_parent():
    """Input_Price_Spike is driven by the regime AND directly by Geo_Exposure, so
    a price spike under high geo is partly explained by macro pricing and
    implicates the regime less (smaller incremental shift)."""
    net = build_network("latent_regime")
    assert set(net.get_parents("Input_Price_Spike")) == {"Disruption_Regime", "Geo_Exposure"}
    ve = VariableElimination(net)

    def p_disrupted(evidence):
        f = ve.query(["Disruption_Regime"], evidence=evidence, show_progress=False)
        d = {s: float(f.values[i]) for i, s in enumerate(f.state_names["Disruption_Regime"])}
        return d["Multi_Node_Ripple"] + d["Severe"]

    delta_low = (p_disrupted({"Geo_Exposure": "Low", "Input_Price_Spike": "Spiking"})
                 - p_disrupted({"Geo_Exposure": "Low"}))
    delta_high = (p_disrupted({"Geo_Exposure": "High", "Input_Price_Spike": "Spiking"})
                  - p_disrupted({"Geo_Exposure": "High"}))
    assert delta_high < delta_low, (delta_high, delta_low)


def test_impact_layer_is_monotone_in_regime():
    """Worse regime ⇒ more mass on the bad impact state (the P&L story)."""
    net = build_network("latent_regime")
    regimes = net.get_cpds("Production_Output").state_names["Disruption_Regime"]
    cpd = net.get_cpds("Production_Output")
    halted = [cpd.get_value(Production_Output="Halted", Disruption_Regime=r) for r in regimes]
    assert halted == sorted(halted), dict(zip(regimes, halted))


def test_headline_evidence_moves_the_margin_node():
    """End-to-end value claim: an emission observation propagates to the
    P&L-relevant Gross_Margin_Hit node, not just the abstract regime."""
    net = build_network("latent_regime")
    ve = VariableElimination(net)

    def heavy(evidence):
        f = ve.query(["Gross_Margin_Hit"], evidence=evidence, show_progress=False)
        return {s: float(f.values[i]) for i, s in enumerate(f.state_names["Gross_Margin_Hit"])}["Heavy"]

    assert heavy({"Lead_Time_Slippage": "Blown", "Force_Majeure_Notices": "Multiple"}) > heavy({})
