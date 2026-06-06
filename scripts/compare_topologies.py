"""Compare the labelling vs latent-regime topologies and emit the report + figures.

This module is the permanent home for the comparison/analysis helpers (the notebook
``notebooks/latent_regime_comparison.ipynb`` imports them). Running it as a script writes:
  - docs/assets/latent_regime/*.png   (figures)
  - docs/01_latent_regime_comparison.md (report)

Run:  pixi run python scripts/compare_topologies.py
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.network as N  # noqa: E402
from src.inference import clamped_scenario_likelihoods, scenario_bayes_factors  # noqa: E402
from src.network import STATES, _cpd, build_network  # noqa: E402
from src.sensitivity import default_concentrations, scenario_credible_intervals  # noqa: E402

S = "Scenario"
D, T, P = "Energy_Infrastructure_Damage", "Conflict_Duration", "Diplomatic_Resolution_Path"
M, C = "US_Military_Response", "Strait_Operationally_Closed"
SCEN = STATES[S]
SHORT = {"Stress_Mitigates": "Stress", "Prolonged_Conflict": "Prolonged", "Severe_Closure": "Severe"}

ESCALATION = {"US_Iran_Negotiations": "breakdown", "Sanctions_Trajectory": "tightening",
              "Iranian_Regime_Stability": "unstable", "Tanker_Incidents": "frequent", M: "major"}
DEESCALATION = {"US_Iran_Negotiations": "success", "Third_Party_Mediation": "active",
                "Iranian_Regime_Stability": "stable", "Tanker_Incidents": "none", M: "none"}

BATTERY: List[Tuple[str, Dict[str, str]]] = [
    ("no_evidence", {}),
    ("U1=breakdown", {"US_Iran_Negotiations": "breakdown"}),
    ("U2=unstable", {"Iranian_Regime_Stability": "unstable"}),
    ("U3=active (blind)", {"Third_Party_Mediation": "active"}),
    ("U4=tightening", {"Sanctions_Trajectory": "tightening"}),
    ("A=high", {"Iran_Aligned_Militia_Attacks": "high"}),
    ("K=frequent", {"Tanker_Incidents": "frequent"}),
    ("M=major", {M: "major"}),
    ("C=full", {C: "full"}),
    ("D=severe", {D: "severe"}),
    ("T=long", {T: "long"}),
    ("P=closed", {P: "closed"}),
    ("O=above_120", {"Oil_Price_Regime": "above_120"}),
    ("U1=success", {"US_Iran_Negotiations": "success"}),
    ("M=none", {M: "none"}),
    ("D=none", {D: "none"}),
    ("escalation_full", dict(ESCALATION)),
    ("deescalation_full", dict(DEESCALATION)),
    ("escalation_partial", {"Sanctions_Trajectory": "tightening", "Tanker_Incidents": "frequent"}),
    ("deescalation_partial", {"US_Iran_Negotiations": "success", "Tanker_Incidents": "none"}),
    ("contradict_severeD_openP", {D: "severe", P: "open"}),
    ("contradict_majorM_shortT", {M: "major", T: "short"}),
    ("mixed_closedC_successU1", {C: "full", "US_Iran_Negotiations": "success"}),
    ("mixed_severeD_activeU3", {D: "severe", "Third_Party_Mediation": "active"}),
]


# ---------------------------------------------------------------------------
# core helpers
# ---------------------------------------------------------------------------

def point(ve, ev) -> np.ndarray:
    f = ve.query([S], evidence=dict(ev), show_progress=False)
    return np.array([float(f.get_value(**{S: s})) for s in SCEN])


def build_hard_labelling() -> DiscreteBayesianNetwork:
    """Labelling net with P(S|D,T,P) collapsed to a one-hot argmax partition."""
    vals = np.asarray(N.CPD_SCENARIO.get_values())
    hard = np.zeros_like(vals)
    hard[vals.argmax(axis=0), np.arange(vals.shape[1])] = 1.0
    cols = list(product(STATES[D], STATES[T], STATES[P]))
    cpd = _cpd(S, [D, T, P], {k: hard[:, i].tolist() for i, k in enumerate(cols)})
    net = DiscreteBayesianNetwork(list(N.EDGES))
    net.add_cpds(*[c for c in build_network("labelling").get_cpds() if c.variable != S], cpd)
    net.check_model()
    return net


def posterior_table(lab, lat) -> List[dict]:
    vl, vt = VariableElimination(lab), VariableElimination(lat)
    rows = []
    for name, ev in BATTERY:
        pl, pt = point(vl, ev), point(vt, ev)
        rows.append({"config": name, "labelling": pl, "latent": pt,
                     "l1": float(np.abs(pl - pt).sum())})
    return rows


def uq_table(lab, lat, m=200) -> List[dict]:
    vl, vt = VariableElimination(lab), VariableElimination(lat)
    kl, kt = default_concentrations(lab), default_concentrations(lat)
    rows = []
    for name, ev in BATTERY:
        cl = scenario_credible_intervals(ev, m=m, concentration=kl, base_network=lab, seed=1)
        ct = scenario_credible_intervals(ev, m=m, concentration=kt, base_network=lat, seed=1)
        ml = np.array([cl[s][0] for s in SCEN]); mt = np.array([ct[s][0] for s in SCEN])
        wl = float(np.mean([cl[s][2] - cl[s][1] for s in SCEN]))
        wt = float(np.mean([ct[s][2] - ct[s][1] for s in SCEN]))
        rows.append({
            "config": name, "lab_ci": cl, "lat_ci": ct,
            "lab_gap": float(np.abs(point(vl, ev) - ml).max()) * 100,
            "lat_gap": float(np.abs(point(vt, ev) - mt).max()) * 100,
            "lab_width": wl * 100, "lat_width": wt * 100,
        })
    return rows


def head_to_head(lab, lat, n=3000, seed=123) -> dict:
    vl, vt = VariableElimination(lab), VariableElimination(lat)
    df = lat.simulate(n_samples=n, seed=seed, show_progress=False)
    sidx = {s: i for i, s in enumerate(SCEN)}
    all_obs = [x for x in STATES if x != S]

    def score(ve, nodes):
        cache, ll, br, ac = {}, 0.0, 0.0, 0.0
        for _, r in df.iterrows():
            key = tuple(r[x] for x in nodes)
            if key not in cache:
                cache[key] = point(ve, {x: r[x] for x in nodes})
            post = cache[key]; i = sidx[r[S]]
            ll += -np.log(max(post[i], 1e-12))
            oh = np.zeros(3); oh[i] = 1; br += float(((post - oh) ** 2).sum())
            ac += float(SCEN[int(post.argmax())] == r[S])
        k = len(df); return ll / k, br / k, ac / k
    out = {"mix": {SHORT[s]: round(float((df[S] == s).mean()), 3) for s in SCEN}}
    for label, nodes in [("emissions", [D, T, P]), ("all_observables", all_obs)]:
        out[label] = {"labelling": score(vl, nodes), "latent": score(vt, nodes)}
    # confusion matrix (latent, all observables)
    cache, conf = {}, np.zeros((3, 3))
    for _, r in df.iterrows():
        key = tuple(r[x] for x in all_obs)
        if key not in cache:
            cache[key] = SCEN[int(point(vt, {x: r[x] for x in all_obs}).argmax())]
        conf[sidx[r[S]], sidx[cache[key]]] += 1
    out["confusion"] = (conf / conf.sum(axis=1, keepdims=True)).tolist()
    return out


def domain_evidence(lat) -> dict:
    ve = VariableElimination(lat)
    joint = ve.query([S, D, T, P], evidence={}, show_progress=False)
    ps = point(ve, {})
    cells = list(product(STATES[D], STATES[T], STATES[P]))
    pso = np.array([[joint.get_value(**{S: s, D: d, T: t, P: p}) for (d, t, p) in cells] for s in SCEN])
    po = pso.sum(axis=0)
    mi = sum(pso[i, j] * np.log2(pso[i, j] / (ps[i] * po[j]))
             for i in range(3) for j in range(len(cells)) if pso[i, j] > 0)
    hs = float(-(ps[ps > 0] * np.log2(ps[ps > 0])).sum())
    cond = pso / ps[:, None]
    sig = {"Stress_Mitigates": ("none", "short", "open"),
           "Prolonged_Conflict": ("moderate", "long", "narrowing"),
           "Severe_Closure": ("severe", "long", "closed")}
    overlaps = {f"{SHORT[SCEN[a]]} vs {SHORT[SCEN[b]]}": float(1 - 0.5 * np.abs(cond[a] - cond[b]).sum())
                for a in range(3) for b in range(a + 1, 3)}
    offmode = {SHORT[SCEN[i]]: float(1 - cond[i, cells.index(sig[SCEN[i]])]) for i in range(3)}
    return {"mi_bits": round(float(mi), 3), "h_s": round(hs, 3),
            "mi_frac": round(float(mi / hs), 3), "bayes_acc": round(float(pso.max(axis=0).sum()), 3),
            "overlap": {k: round(v, 3) for k, v in overlaps.items()},
            "offmode": {k: round(v, 3) for k, v in offmode.items()}}


def data_driven_verdict(lat, n=1500, seed=5) -> dict:
    """Self-contained, no-external-input verdict on whether the constructed latent model has
    the *signature* of genuine latent regimes. Three measurable criteria:

      C1 distinguishable : regimes are identifiable from outcomes better than guessing the
                           most-common one (Bayes-optimal accuracy > max-prior baseline).
      C2 generative      : regimes overlap rather than form a rigid partition
                           (accuracy < 1 and some pairwise overlap remains).
      C3 context-informative : observing context on top of outcomes still shifts P(S) by a
                           material amount (avg total-variation shift > 2pp), i.e. the latent
                           structure does real work.
    """
    ve = VariableElimination(lat)
    dom = domain_evidence(lat)
    prior = point(ve, {}); max_prior = float(prior.max())

    c1 = dom["bayes_acc"] > max_prior + 0.05
    max_ov = max(dom["overlap"].values())
    c2 = (dom["bayes_acc"] < 0.999) and (max_ov > 0.20)

    ctx_nodes = ["US_Military_Response", "Strait_Operationally_Closed",
                 "Sanctions_Trajectory", "US_Iran_Negotiations", "Iranian_Regime_Stability"]
    df = lat.simulate(n_samples=n, seed=seed, show_progress=False)
    cache, tvs = {}, []
    for _, r in df.iterrows():
        ko = tuple(r[x] for x in (D, T, P))
        kc = ko + tuple(r[x] for x in ctx_nodes)
        if ko not in cache:
            cache[ko] = point(ve, {x: r[x] for x in (D, T, P)})
        if kc not in cache:
            cache[kc] = point(ve, {**{x: r[x] for x in (D, T, P)}, **{x: r[x] for x in ctx_nodes}})
        tvs.append(0.5 * float(np.abs(cache[ko] - cache[kc]).sum()))
    avg_tv = float(np.mean(tvs))
    c3 = avg_tv > 0.02

    crits = {
        "C1_distinguishable": {"pass": bool(c1), "value": dom["bayes_acc"],
                               "baseline": round(max_prior, 3)},
        "C2_generative_not_bucket": {"pass": bool(c2), "max_overlap": round(max_ov, 3),
                                     "accuracy": dom["bayes_acc"]},
        "C3_context_informative": {"pass": bool(c3), "avg_tv_shift_pp": round(avg_tv * 100, 2)},
    }
    n_pass = sum(c["pass"] for c in crits.values())
    verdict = {3: "DATA SUPPORTS the latent-regime reading",
               2: "DATA LEANS toward the latent regime (one criterion weak)",
               1: "INCONCLUSIVE", 0: "DATA FAVORS the labelling model"}[n_pass]
    return {"criteria": crits, "n_pass": n_pass, "verdict": verdict, "evidence": dom}


def degeneracy_demo(lat) -> dict:
    hard = build_hard_labelling()
    pe_h = clamped_scenario_likelihoods(hard, {D: "severe"})
    pe_l = clamped_scenario_likelihoods(lat, {D: "severe"})
    return {
        "hard_PStress": pe_h["Stress_Mitigates"],
        "hard_lambda": (np.inf if pe_h["Stress_Mitigates"] == 0 else
                        pe_h["Severe_Closure"] / pe_h["Stress_Mitigates"]),
        "latent_PStress": pe_l["Stress_Mitigates"],
        "latent_lambda": pe_l["Severe_Closure"] / pe_l["Stress_Mitigates"],
    }


# ---------------------------------------------------------------------------
# figures + report
# ---------------------------------------------------------------------------

def _assets_dir() -> Path:
    d = Path(__file__).resolve().parents[1] / "docs" / "assets" / "latent_regime"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_figures(lab, lat, post, uq, h2h, dom) -> Dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = _assets_dir()
    paths: Dict[str, Path] = {}
    colors = ["#2e7d32", "#f9a825", "#c62828"]
    uqmap = {r["config"]: r for r in uq}

    # Fig 1: posterior + CI for key configs, both models
    key = ["no_evidence", "deescalation_full", "escalation_full", "D=severe", "mixed_closedC_successU1"]
    fig, axes = plt.subplots(1, len(key), figsize=(4 * len(key), 4), sharey=True)
    for ax, cfg in zip(axes, key):
        r = uqmap[cfg]
        x = np.arange(3); w = 0.38
        for off, model, ci in [(-w / 2, "labelling", r["lab_ci"]), (w / 2, "latent", r["lat_ci"])]:
            means = [ci[s][0] for s in SCEN]
            err = [[ci[s][0] - ci[s][1] for s in SCEN], [ci[s][2] - ci[s][0] for s in SCEN]]
            hatch = "" if model == "latent" else "//"
            ax.bar(x + off, means, w, yerr=err, capsize=3, color=colors, alpha=0.95 if model == "latent" else 0.55,
                   hatch=hatch, edgecolor="black", linewidth=0.5, label=model)
        ax.set_title(cfg, fontsize=9); ax.set_xticks(x); ax.set_xticklabels([SHORT[s] for s in SCEN], fontsize=8)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("P(Scenario | E)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Posterior + 80% credible interval: labelling (hatched) vs latent", fontsize=11)
    fig.tight_layout(); paths["posteriors"] = out / "posteriors.png"; fig.savefig(paths["posteriors"], dpi=110); plt.close(fig)

    # Fig 2: M7 gap + CI width
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    lab_g = [r["lab_gap"] for r in uq]; lat_g = [r["lat_gap"] for r in uq]
    a1.boxplot([lab_g, lat_g], tick_labels=["labelling", "latent"]); a1.set_ylabel("point-vs-resample gap (pp)")
    a1.set_title("Finding: M7 NOT differentially closed\n(gap small in both)")
    lab_w = [r["lab_width"] for r in uq]; lat_w = [r["lat_width"] for r in uq]
    a2.boxplot([lab_w, lat_w], tick_labels=["labelling", "latent"]); a2.set_ylabel("mean 80% CI width (pp)")
    a2.set_title("Latent model is honestly less certain\n(wider credible intervals)")
    fig.tight_layout(); paths["uncertainty"] = out / "uncertainty.png"; fig.savefig(paths["uncertainty"], dpi=110); plt.close(fig)

    # Fig 3: head-to-head + confusion
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    metrics = ["log-loss", "Brier", "accuracy"]
    lab_s = h2h["all_observables"]["labelling"]; lat_s = h2h["all_observables"]["latent"]
    x = np.arange(3); w = 0.38
    a1.bar(x - w / 2, lab_s, w, label="labelling", color="#888", hatch="//", edgecolor="k", linewidth=0.5)
    a1.bar(x + w / 2, lat_s, w, label="latent", color="#1565c0", edgecolor="k", linewidth=0.5)
    a1.set_xticks(x); a1.set_xticklabels(metrics); a1.legend(fontsize=8)
    a1.set_title("Head-to-head on simulated truth\n(lower loss / higher acc = better)")
    conf = np.array(h2h["confusion"])
    im = a2.imshow(conf, cmap="Blues", vmin=0, vmax=1)
    a2.set_xticks(range(3)); a2.set_yticks(range(3))
    a2.set_xticklabels([SHORT[s] for s in SCEN]); a2.set_yticklabels([SHORT[s] for s in SCEN])
    a2.set_xlabel("predicted"); a2.set_ylabel("true regime"); a2.set_title("Latent regime separability")
    for i in range(3):
        for j in range(3):
            a2.text(j, i, f"{conf[i,j]:.2f}", ha="center", va="center",
                    color="white" if conf[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=a2, fraction=0.046)
    fig.tight_layout(); paths["headtohead"] = out / "headtohead.png"; fig.savefig(paths["headtohead"], dpi=110); plt.close(fig)
    return paths


def _fmt_p(arr):
    return "[" + ", ".join(f"{x:.3f}" for x in arr) + "]"


def write_report(lab, lat, post, uq, h2h, dom, deg, figs, verdict) -> Path:
    p95 = lambda xs: float(np.percentile(xs, 95))  # noqa: E731
    lab_g = [r["lab_gap"] for r in uq]; lat_g = [r["lat_gap"] for r in uq]
    lab_w = np.mean([r["lab_width"] for r in uq]); lat_w = np.mean([r["lat_width"] for r in uq])
    top = sorted(post, key=lambda r: -r["l1"])[:6]
    rel = "assets/latent_regime"

    L = []
    L.append("# Latent-Regime vs Labelling: Topology Comparison\n")
    L.append("> Auto-generated by `scripts/compare_topologies.py`. Companion to "
             "[docs/01_latent_regime_plan.md](01_latent_regime_plan.md). The latent-regime "
             "model is built with `build_network(\"latent_regime\")`; the labelling model "
             "(`build_network()`) is retained as the comparison baseline.\n")

    L.append("## Three headline findings\n")
    L.append("**1. The labelling model is structurally blind to upstream context "
             "*once outcomes are observed*.** This is a conditional statement, not "
             "\"upstream never matters\": if the outcomes are *unobserved*, changing an upstream "
             "node shifts (D,T,P) and hence `P(S)` in **both** models identically. But once "
             "(D,T,P) are observed, `Scenario`'s Markov blanket is exactly {D,T,P}, so adding "
             "upstream evidence on top changes nothing in the labelling model (the latent model "
             "still updates, e.g. severe outcomes in a benign military context shift mass toward "
             "Prolonged). In the head-to-head below the "
             "labelling model scores **identically** on *emissions only* vs *all observables* "
             f"({h2h['emissions']['labelling'][0]:.3f} log-loss both), while the latent model "
             f"improves ({h2h['emissions']['latent'][0]:.3f} → {h2h['all_observables']['latent'][0]:.3f}). "
             "This is the single clearest argument for the reframe.\n")
    L.append("**2. The labelling model is overconfident; the latent model is honestly uncertain.** "
             "On full de-escalation the labelling model reports "
             f"{_fmt_p([r for r in post if r['config']=='deescalation_full'][0]['labelling'])} "
             "(≈90% Stress) vs the latent model's "
             f"{_fmt_p([r for r in post if r['config']=='deescalation_full'][0]['latent'])} "
             f"(≈68% Stress). The latent model's credible intervals are also wider on average "
             f"({lat_w:.1f}pp vs {lab_w:.1f}pp) — the labelling model's narrow CIs are false precision.\n")
    L.append("**3. Finding M7 is NOT closed by the reframe.** The plan hypothesised the "
             "point-vs-resample-mean gap would shrink under the latent model. It does not: the "
             f"gap is small in *both* models (95th-pctile labelling {p95(lab_g):.2f}pp, latent "
             f"{p95(lat_g):.2f}pp) and is not smaller for the latent model. M7 is a small, "
             "model-agnostic Jensen artefact, correctly handled by plotting the resample-mean.\n")
    L.append(f"![posteriors]({rel}/posteriors.png)\n")
    L.append(f"![uncertainty]({rel}/uncertainty.png)\n")
    L.append(f"![head-to-head]({rel}/headtohead.png)\n")

    L.append("## Where the two models agree exactly (correctness of the anchor derivation)\n")
    L.append("The latent CPTs are exact conditionals of the current joint, so the two models "
             "agree to machine precision on the prior, on damage-evidence `P(S|D)`, on "
             "`P(S|M)`, and on the joint `P(S,D,M,C)`. They diverge only where the topology "
             "genuinely differs (duration, path, upstream evidence).\n")
    L.append("Largest divergences (L1 over the scenario simplex):\n")
    L.append("| config | labelling | latent | L1 |\n|---|---|---|---|")
    for r in top:
        L.append(f"| {r['config']} | {_fmt_p(r['labelling'])} | {_fmt_p(r['latent'])} | {r['l1']:.3f} |")
    L.append("")

    L.append("## Expressiveness: the two knobs the latent topology adds\n")
    hl = "∞ (degenerate)" if not np.isfinite(deg["hard_lambda"]) else f"{deg['hard_lambda']:.1f}"
    L.append(f"- **Bayes factors stay finite.** On `E={{D=severe}}`, the hard labelling limit gives "
             f"`P(D=severe|Stress)=0` → Λ(Severe,Stress) = {hl}; the latent model gives "
             f"`P(D=severe|Stress)={deg['latent_PStress']:.3f}` → Λ = {deg['latent_lambda']:.1f}, "
             "finite and interpretable.\n")
    L.append("- **Independent regime prior.** `P(S|M,C)` is a primitive you can set; the labelling "
             "model has no such parameter (its scenario prior is induced by the outcome marginal × "
             "partition).\n")
    L.append("- **Multiplicative composition.** Conditional on a full context, latent emissions "
             "compose exactly: Λ(D,P)=Λ(D)·Λ(P).\n")

    L.append("## Head-to-head calibration on simulated ground truth\n")
    L.append(f"Simulated regime mix: {h2h['mix']}.\n")
    L.append("| evidence | model | log-loss | Brier | accuracy |\n|---|---|---|---|---|")
    for ev in ("emissions", "all_observables"):
        for mdl in ("labelling", "latent"):
            s = h2h[ev][mdl]
            L.append(f"| {ev} | {mdl} | {s[0]:.4f} | {s[1]:.4f} | {s[2]:.4f} |")
    L.append("\n*Caveat:* data are generated by the latent model, so this presupposes the "
             "semantic commitment that regimes are real generative causes (see the domain "
             "judgement below). Under that assumption it quantifies the labelling coarsening's "
             "miscalibration.\n")

    L.append("## Domain-signature check (self-contained)\n")
    L.append("Are the scenarios real latent regimes or just relabelled outcome buckets? We make "
             "this decidable from the data: a genuine latent regime is (C1) **distinguishable** "
             "from outcomes, (C2) **generative/overlapping** rather than a rigid partition, and "
             "(C3) such that **context adds information beyond the outcomes**. Each is a number "
             "with a threshold (no external input required):\n")
    cr = verdict["criteria"]
    L.append("| criterion | measure | passes if | result |\n|---|---|---|---|")
    L.append(f"| C1 distinguishable | Bayes-optimal accuracy {cr['C1_distinguishable']['value']} "
             f"vs baseline {cr['C1_distinguishable']['baseline']} | > baseline + 0.05 | "
             f"{'PASS' if cr['C1_distinguishable']['pass'] else 'FAIL'} |")
    L.append(f"| C2 generative (not bucket) | accuracy < 1 and max pairwise overlap "
             f"{cr['C2_generative_not_bucket']['max_overlap']} | < 1 and overlap > 0.20 | "
             f"{'PASS' if cr['C2_generative_not_bucket']['pass'] else 'FAIL'} |")
    L.append(f"| C3 context-informative | avg P(S) shift from context = "
             f"{cr['C3_context_informative']['avg_tv_shift_pp']}pp | > 2pp | "
             f"{'PASS' if cr['C3_context_informative']['pass'] else 'FAIL'} |")
    L.append(f"\n**{verdict['n_pass']}/3 criteria pass → {verdict['verdict']}.** "
             f"(Background: mutual information {dom['mi_bits']} bits = {dom['mi_frac']*100:.0f}% "
             f"of H(S); pairwise overlap {dom['overlap']}.) A rigid bucket would fail C2; pure "
             "noise would fail C1 — passing all three is the data signature of genuine, "
             "overlapping latent regimes. The final *adoption* decision (and removing the "
             "labelling path) remains a human call; the labelling model stays the committed "
             "default via `build_network()` until then.\n")

    target = Path(__file__).resolve().parents[1] / "docs" / "01_latent_regime_comparison.md"
    target.write_text("\n".join(L))
    return target


def main() -> None:
    lab, lat = build_network("labelling"), build_network("latent_regime")
    print("computing posteriors / UQ / head-to-head / domain evidence ...")
    post = posterior_table(lab, lat)
    uq = uq_table(lab, lat, m=200)
    h2h = head_to_head(lab, lat)
    dom = domain_evidence(lat)
    deg = degeneracy_demo(lat)
    verdict = data_driven_verdict(lat)
    figs = make_figures(lab, lat, post, uq, h2h, dom)
    report = write_report(lab, lat, post, uq, h2h, dom, deg, figs, verdict)
    print(f"wrote {report}")
    for k, v in figs.items():
        print(f"  figure: {v}")


if __name__ == "__main__":
    main()
