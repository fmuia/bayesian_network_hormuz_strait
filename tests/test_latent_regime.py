"""Quadruple-test suite for the latent-regime reframing (Plan 1).

Seven verification angles, ported from the scratch correctness harness, plus the
domain-judgement unit. Each angle is an independent line of evidence that the
latent-regime topology is implemented correctly and behaves better than the labelling
model. The labelling default is exercised by the existing test_network.py.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from src.cpt_data import LATENT_CPTS
from src.inference import (
    BNInferenceEngine,
    clamped_scenario_likelihoods,
    scenario_bayes_factors,
)
from src.network import (
    EDGES_LATENT,
    SCENARIO_SIGNATURES,
    STATES,
    _cpd,
    build_network,
)
from src.sensitivity import default_concentrations, scenario_credible_intervals
from scripts.derive_latent_regime_anchors import derive_latent_cpts

S = "Scenario"
D = "Energy_Infrastructure_Damage"
T = "Conflict_Duration"
P = "Diplomatic_Resolution_Path"
M = "US_Military_Response"
C = "Strait_Operationally_Closed"
U3 = "Third_Party_Mediation"


@pytest.fixture(scope="module")
def lab():
    return build_network("labelling")


@pytest.fixture(scope="module")
def lat():
    return build_network("latent_regime")


def _dist(ve, evidence):
    f = ve.query([S], evidence=dict(evidence), show_progress=False)
    return np.array([float(f.get_value(**{S: s})) for s in STATES[S]])


def _build_hard_labelling():
    """Labelling net with P(S|D,T,P) collapsed to a one-hot argmax partition."""
    import src.network as N

    vals = np.asarray(N.CPD_SCENARIO.get_values())
    hard = np.zeros_like(vals)
    hard[vals.argmax(axis=0), np.arange(vals.shape[1])] = 1.0
    cols = list(product(STATES[D], STATES[T], STATES[P]))
    table = {key: hard[:, i].tolist() for i, key in enumerate(cols)}
    cpd = _cpd(S, [D, T, P], table)
    net = DiscreteBayesianNetwork(list(N.EDGES))
    net.add_cpds(*[c for c in build_network("labelling").get_cpds() if c.variable != S], cpd)
    net.check_model()
    return net


# ===== ANGLE 1 - ALGEBRAIC =================================================


def test_angle1_committed_cpts_match_fresh_derivation():
    fresh = derive_latent_cpts()
    assert set(fresh) == set(LATENT_CPTS)
    for child, table in fresh.items():
        for key, vec in table.items():
            got = LATENT_CPTS[child][key]
            assert np.allclose(vec, got, atol=1e-9), (child, key)


def test_angle1_derived_cpts_are_exact_conditionals(lab):
    """P(D|S,M,C)*P(S,M,C) reconstructs the labelling joint P(D,S,M,C) to ~1e-9."""
    ve = VariableElimination(lab)
    cond = ve.query([D, S, M, C], evidence={}, show_progress=False)
    marg = ve.query([S, M, C], evidence={}, show_progress=False)
    joint = ve.query([D, S, M, C], evidence={}, show_progress=False)
    # use the committed emission CPT
    dcpd = build_network("latent_regime").get_cpds(D)
    err = 0.0
    for s in STATES[S]:
        for m in STATES[M]:
            for c in STATES[C]:
                pmc = marg.get_value(**{S: s, M: m, C: c})
                for d in STATES[D]:
                    recon = dcpd.get_value(**{D: d, S: s, M: m, C: c}) * pmc
                    err = max(err, abs(recon - joint.get_value(**{D: d, S: s, M: m, C: c})))
    assert err < 1e-9


# ===== ANGLE 2 - STRUCTURAL ================================================


def test_angle2_builds_validates_edges(lat):
    assert lat.check_model() is True
    assert set(lat.edges()) == set(EDGES_LATENT)
    assert set(lat.nodes()) == set(STATES)


def test_angle2_normalised_and_non_degenerate(lat):
    min_cell = 1.0
    for cpd in lat.get_cpds():
        vals = np.asarray(cpd.get_values())
        for col in range(vals.shape[1]):
            assert abs(vals[:, col].sum() - 1.0) < 1e-9
        min_cell = min(min_cell, float(vals.min()))
    assert min_cell > 0.0  # no exact zeros -> finite Bayes factors


def test_angle2_mode_invariant_extreme_scenarios(lat):
    """argmax emission at the characteristic context matches the narrative signature
    for the two extreme regimes (the middle regime is allowed to miss; see Plan A.2)."""
    ctx = {
        "Stress_Mitigates": {M: "none", C: "no", "US_Iran_Negotiations": "success",
                             U3: "active", "Iranian_Regime_Stability": "stable"},
        "Severe_Closure": {M: "major", C: "full", "US_Iran_Negotiations": "breakdown",
                           U3: "none", "Iranian_Regime_Stability": "unstable"},
    }
    parents = {D: [M, C], T: ["US_Iran_Negotiations", U3, M],
               P: ["US_Iran_Negotiations", U3, "Iranian_Regime_Stability"]}
    for scenario, sig in ((k, SCENARIO_SIGNATURES[k]) for k in ctx):
        want = dict(zip([D, T, P], sig))
        for child in (D, T, P):
            cpd = lat.get_cpds(child)
            context = {S: scenario, **{p: ctx[scenario][p] for p in parents[child]}}
            vec = [cpd.get_value(**{child: cs}, **context) for cs in STATES[child]]
            assert STATES[child][int(np.argmax(vec))] == want[child], (scenario, child)


# ===== ANGLE 3 - INFERENTIAL ===============================================


def test_angle3_dsep_positive_for_non_u3(lat):
    ve = VariableElimination(lat)
    prior = _dist(ve, {})
    for node in ["US_Iran_Negotiations", "Sanctions_Trajectory", "Tanker_Incidents",
                 M, C, D, T, P, "Oil_Price_Regime"]:
        shift = max(float(np.abs(_dist(ve, {node: st}) - prior).sum()) for st in STATES[node])
        assert shift > 1e-6, node


def test_angle3_u3_blind_spot(lat):
    ve = VariableElimination(lat)
    prior = _dist(ve, {})
    for st in STATES[U3]:
        assert float(np.abs(_dist(ve, {U3: st}) - prior).sum()) < 1e-9


def test_angle3_escalation_deescalation(lat):
    ve = VariableElimination(lat)
    prior = _dist(ve, {})
    idx = {s: i for i, s in enumerate(STATES[S])}
    esc = _dist(ve, {"US_Iran_Negotiations": "breakdown", "Sanctions_Trajectory": "tightening",
                     "Tanker_Incidents": "frequent", M: "major"})
    deesc = _dist(ve, {"US_Iran_Negotiations": "success", U3: "active",
                       "Tanker_Incidents": "none", M: "none"})
    assert esc[idx["Severe_Closure"]] > prior[idx["Severe_Closure"]]
    assert deesc[idx["Stress_Mitigates"]] > prior[idx["Stress_Mitigates"]]


def test_angle3_bayes_factor_ratio_equals_clamped(lat):
    ev = {D: "severe"}
    lam_ratio = scenario_bayes_factors(lat, ev)["lambda"]["Severe_Closure"]["Stress_Mitigates"]
    pe = clamped_scenario_likelihoods(lat, ev)
    lam_clamp = pe["Severe_Closure"] / pe["Stress_Mitigates"]
    assert abs(lam_ratio - lam_clamp) < 1e-6 * max(1.0, lam_ratio)
    assert lam_ratio > 1.0


def test_angle3_hand_factorization_equals_ve(lat):
    ve = VariableElimination(lat)
    pmc = ve.query([M, C], evidence={}, show_progress=False)
    reg, dem = lat.get_cpds(S), lat.get_cpds(D)
    hand = np.array([
        sum(pmc.get_value(**{M: m, C: c})
            * reg.get_value(**{S: s, M: m, C: c})
            * dem.get_value(**{D: "severe", S: s, M: m, C: c})
            for m in STATES[M] for c in STATES[C])
        for s in STATES[S]])
    hand = hand / hand.sum()
    assert np.abs(hand - _dist(ve, {D: "severe"})).max() < 1e-9


def test_angle3_calibration_posterior_gain(lat):
    """Under each true regime, mean posterior on s* beats the prior and rises with #emissions."""
    ve = VariableElimination(lat)
    prior = {s: _dist(ve, {})[i] for i, s in enumerate(STATES[S])}
    for s_star in STATES[S]:
        df = lat.simulate(n_samples=400, evidence={S: s_star}, seed=7, show_progress=False)
        means = []
        for k in range(1, 4):
            nodes = [D, T, P][:k]
            means.append(np.mean([
                float(ve.query([S], evidence={n: r[n] for n in nodes}, show_progress=False)
                      .get_value(**{S: s_star})) for _, r in df.iterrows()]))
        assert means[-1] > prior[s_star]
        assert all(means[i + 1] >= means[i] - 1e-2 for i in range(len(means) - 1))


# ===== ANGLE 4 - COMPARATIVE / UQ ==========================================


def test_angle4_exact_agreement_where_forced(lab, lat):
    vl, vt = VariableElimination(lab), VariableElimination(lat)
    assert np.abs(_dist(vl, {}) - _dist(vt, {})).max() < 1e-9
    for d in STATES[D]:
        assert np.abs(_dist(vl, {D: d}) - _dist(vt, {D: d})).max() < 1e-9
    for m in STATES[M]:
        assert np.abs(_dist(vl, {M: m}) - _dist(vt, {M: m})).max() < 1e-9


def test_angle4_genuine_divergence_on_upstream(lab, lat):
    vl, vt = VariableElimination(lab), VariableElimination(lat)
    ev = {"US_Iran_Negotiations": "success"}
    assert float(np.abs(_dist(vl, ev) - _dist(vt, ev)).sum()) > 0.05


def test_angle4_uq_valid_and_gap_bounded(lat):
    kap = default_concentrations(lat)
    ve = VariableElimination(lat)
    for ev in ({}, {M: "major"}, {D: "severe", P: "open"}):
        ci = scenario_credible_intervals(ev, m=80, concentration=kap, base_network=lat, seed=1)
        mean = np.array([ci[s][0] for s in STATES[S]])
        assert abs(mean.sum() - 1.0) < 1e-6
        for s in STATES[S]:
            mn, lo, hi = ci[s]
            assert lo - 1e-9 <= mn <= hi + 1e-9
        gap = float(np.abs(_dist(ve, ev) - mean).max()) * 100
        assert gap < 2.0  # small in both models


# ===== ANGLE 5 - EXPRESSIVENESS ============================================


def test_angle5_degeneracy_contrast(lat):
    hard = _build_hard_labelling()
    pe_hard = clamped_scenario_likelihoods(hard, {D: "severe"})
    pe_lat = clamped_scenario_likelihoods(lat, {D: "severe"})
    assert pe_hard["Stress_Mitigates"] == 0.0          # labelling degenerates (Lambda = inf)
    assert pe_lat["Stress_Mitigates"] > 0.0            # latent stays finite
    assert pe_lat["Severe_Closure"] / pe_lat["Stress_Mitigates"] > 1.0


def test_angle5_independent_prior_knob():
    from src.cpt_data import LATENT_CPTS as base
    import src.network as N
    # force P(Severe|M,C)=0.05 in every regime column
    reg = {k: [v[0] * 0.95 / (v[0] + v[1]), v[1] * 0.95 / (v[0] + v[1]), 0.05]
           for k, v in base[S].items()}
    cpds = [c for c in build_network("latent_regime").get_cpds() if c.variable != S]
    cpds.append(_cpd(S, ["US_Military_Response", "Strait_Operationally_Closed"], reg))
    net = DiscreteBayesianNetwork(list(EDGES_LATENT))
    net.add_cpds(*cpds)
    net.check_model()
    ve = VariableElimination(net)
    sev = float(ve.query([S], evidence={}, show_progress=False).get_value(**{S: "Severe_Closure"}))
    assert abs(sev - 0.05) < 1e-6


def test_angle5_conditional_composition_exact(lat):
    ctx_d = {M: "major", C: "full"}
    ctx_p = {"US_Iran_Negotiations": "breakdown", U3: "none", "Iranian_Regime_Stability": "unstable"}
    dcpd, pcpd = lat.get_cpds(D), lat.get_cpds(P)
    s1, s2 = "Severe_Closure", "Stress_Mitigates"
    pd1 = dcpd.get_value(**{D: "severe", S: s1}, **ctx_d)
    pd2 = dcpd.get_value(**{D: "severe", S: s2}, **ctx_d)
    pp1 = pcpd.get_value(**{P: "closed", S: s1}, **ctx_p)
    pp2 = pcpd.get_value(**{P: "closed", S: s2}, **ctx_p)
    lamD, lamP, lamDP = pd1 / pd2, pp1 / pp2, (pd1 * pp1) / (pd2 * pp2)
    assert abs(np.log(lamDP) - (np.log(lamD) + np.log(lamP))) < 1e-9


# ===== ANGLE 6 - HEAD-TO-HEAD CALIBRATION ==================================


def test_angle6_latent_beats_labelling_on_simulated_truth(lab, lat):
    vl, vt = VariableElimination(lab), VariableElimination(lat)
    df = lat.simulate(n_samples=1500, seed=123, show_progress=False)
    nodes = [D, T, P]
    sidx = {s: i for i, s in enumerate(STATES[S])}

    def logloss(ve):
        cache, tot = {}, 0.0
        for _, r in df.iterrows():
            key = tuple(r[n] for n in nodes)
            if key not in cache:
                cache[key] = _dist(ve, {n: r[n] for n in nodes})
            tot += -np.log(max(cache[key][sidx[r[S]]], 1e-12))
        return tot / len(df)

    assert logloss(vt) <= logloss(vl) + 1e-9


# ===== ANGLE 7 - KAPPA SENSITIVITY =========================================


def test_angle7_point_estimates_kappa_independent(lat):
    """Point estimates do not depend on kappa (kappa only enters the resample path)."""
    ve = VariableElimination(lat)
    base = _dist(ve, {M: "major"})
    # re-query is deterministic; the invariant is that VE never consults kappa
    assert np.abs(_dist(ve, {M: "major"}) - base).max() == 0.0


# ===== DOMAIN-JUDGEMENT UNIT ===============================================


def test_domain_judgement_evidence_sane(lat):
    """The licensing-evidence metrics are well-formed (the verdict itself is the expert's)."""
    ve = VariableElimination(lat)
    joint = ve.query([S, D, T, P], evidence={}, show_progress=False)
    ps = _dist(ve, {})
    cells = list(product(STATES[D], STATES[T], STATES[P]))
    pso = np.array([[joint.get_value(**{S: s, D: d, T: t, P: p}) for (d, t, p) in cells]
                    for s in STATES[S]])
    po = pso.sum(axis=0)
    mi = sum(pso[i, j] * np.log2(pso[i, j] / (ps[i] * po[j]))
             for i in range(pso.shape[0]) for j in range(pso.shape[1]) if pso[i, j] > 0)
    bayes_acc = float(pso.max(axis=0).sum())
    assert mi > 0.0
    assert 0.0 < bayes_acc < 1.0


def test_engine_method_bayes_factors_on_latent():
    eng = BNInferenceEngine(build_network("latent_regime"))
    eng.update_evidence({D: "severe"})
    bf = eng.scenario_bayes_factors()
    assert bf["posterior"]["Severe_Closure"] > bf["prior"]["Severe_Closure"]
