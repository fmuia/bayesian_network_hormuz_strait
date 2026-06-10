"""Streamlit dashboard for the Strait of Hormuz BN demo.

Run with: ``streamlit run app/dashboard.py`` (or ``pixi run app``).

Layout:

    Sidebar         : provider chip + day controls + headline input
                      + compact single-line translator stream + examples + reset
    Pinned header   : scenario cards + probability evolution (Altair)
    Tabs            : Network & model / Observations / Audit trail

The Network tab uses streamlit-agraph so nodes are clickable; clicking
a node opens an inline "override" form on the right, replacing the
previous sidebar manual picker.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.inference import BNInferenceEngine
from src.network import STATES, build_network
from src.sensitivity import (
    node_credible_intervals,
    scenario_credible_intervals,
)
from src.elicitation.export import spec_from_dict

# Ensure sibling modules (elicitation_panel) import under both `streamlit run`
# and the test harness.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import elicitation_panel  # noqa: E402
import state  # noqa: E402  (sibling module; relies on the sys.path insert above)
from state import current_evidence as _merged_evidence  # noqa: E402

# ---------------------------------------------------------------------------
# Page setup & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Strait of Hormuz — Adaptive Scenario Probabilities",
    layout="wide",
    initial_sidebar_state="expanded",
)

from components import (  # noqa: E402
    audit_view, edge_rationale, evolution_chart, model_explainer, network_view,
    observation_log, scenario_cards, translator_stream, triage_view,
)

# Styles live in app/styles.css (Plan 5 P1 / A2, V8). Loaded once at startup so
# the stylesheet is editable without touching Python.
def _inject_styles() -> None:
    css = (Path(__file__).resolve().parent / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


_inject_styles()


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
# The dashboard runs on the Plan 1 latent-regime topology (Scenario is a latent
# cause generating the outcomes). The labelling model remains available via
# build_network("labelling") for the comparison notebook/scripts, but is not
# surfaced here.
TOPOLOGY = "latent_regime"


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource
def _bootstrap_engine(topology: str = TOPOLOGY) -> BNInferenceEngine:
    return BNInferenceEngine(build_network(topology))


def _locked_spec_json() -> str:
    """The locked elicited network as a JSON string, or '' for the bootstrap."""
    return st.session_state.get("locked_spec_json", "") or ""


def _network_and_concentration(topology: str, locked_spec_json: str):
    """(base_network, concentration) for the active config. A locked elicitation
    wins (its per-CPT kappa map, defaulting other nodes to 20); otherwise the
    bootstrap network for the selected topology at scalar kappa=20."""
    if locked_spec_json:
        spec = spec_from_dict(json.loads(locked_spec_json))
        net = spec.to_pgmpy()
        km = spec.kappa_map()
        return net, {v: km.get(v, 20.0) for v in net.nodes()}
    return build_network(topology), 20.0


def get_engine(topology: str = TOPOLOGY) -> BNInferenceEngine:
    """The inference engine for the active network: a locked elicitation when one
    is set (Plan 4), else the bootstrap network for the selected topology."""
    locked = _locked_spec_json()
    if locked:
        return BNInferenceEngine(spec_from_dict(json.loads(locked)).to_pgmpy())
    return _bootstrap_engine(topology)


@st.cache_data(show_spinner=False)
def cached_credible_intervals(
    evidence_items: Tuple[Tuple[str, str], ...],
    topology: str = TOPOLOGY,
    locked_spec_json: str = "",
) -> Dict[str, Tuple[float, float, float]]:
    base, concentration = _network_and_concentration(topology, locked_spec_json)
    return scenario_credible_intervals(
        dict(evidence_items), m=200, concentration=concentration, base_network=base
    )


@st.cache_data(show_spinner="Computing node uncertainty…")
def cached_node_credible_intervals(
    evidence_items: Tuple[Tuple[str, str], ...],
    soft_evidence_items: Tuple[Tuple[str, Tuple[Tuple[str, float], ...]], ...],
    topology: str = TOPOLOGY,
    locked_spec_json: str = "",
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    soft = {node: dict(dist) for node, dist in soft_evidence_items}
    base, concentration = _network_and_concentration(topology, locked_spec_json)
    return node_credible_intervals(
        dict(evidence_items),
        soft_evidence=soft,
        m=200,
        concentration=concentration,
        base_network=base,
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

state.init_session_state()


translator_stream.render_sidebar(st)

# Plan 4 elicitation layer: run/load an elicitation, inspect reasonings & scores,
# override columns, and lock it — the rest of the dashboard then runs on it.
with st.expander(
    "🧪 Elicitation layer — run / load / override / lock the CPTs", expanded=False
):
    elicitation_panel.render(st, topology=TOPOLOGY)

engine = get_engine(TOPOLOGY)
engine.clear_evidence()
evidence, soft_evidence = _merged_evidence()
# Scenario is the latent regime we infer, never observe — drop any evidence on it
# (e.g. a mistaken manual override) so the Scenario-targeted CI query stays valid.
evidence.pop("Scenario", None)
soft_evidence.pop("Scenario", None)
if evidence:
    engine.update_evidence(evidence)
if soft_evidence:
    engine.update_soft_evidence(soft_evidence)

scenario_probs = engine.get_scenario_probabilities()
ci_evidence = dict(evidence)
for node, dist in soft_evidence.items():
    ci_evidence[node] = max(dist, key=dist.get)
with st.spinner("Quantifying parameter uncertainty…"):
    ci_table = cached_credible_intervals(
        tuple(sorted(ci_evidence.items())), TOPOLOGY, _locked_spec_json()
    )

all_marginals = {n: engine.get_node_marginal(n) for n in STATES}

soft_evidence_ci_items = tuple(
    (node, tuple(sorted(dist.items())))
    for node, dist in sorted(soft_evidence.items())
)
node_ci_table = cached_node_credible_intervals(
    tuple(sorted(evidence.items())),
    soft_evidence_ci_items,
    TOPOLOGY,
    _locked_spec_json(),
)

# Map each observed node to the latest observation that set it (day for the agraph
# payload; full meta — day / source / headline — for the observed-node panel, P8).
observed_day_map: Dict[str, int] = {}
observed_meta: Dict[str, dict] = {}
for obs in st.session_state.observations:
    for node in {**obs.get("assignments", {}), **obs.get("soft_assignments", {})}:
        observed_day_map[node] = obs["day"]
        observed_meta[node] = {"day": obs["day"], "source": obs["source"],
                               "headline": obs["headline"]}

# Standalone Bayes-factor contribution of the selected observed node (P8 / C4):
# what that single observation alone says about the latent regime — hard (a state
# pin) or soft (the translator's ε vector). Only meaningful on a latent-regime
# network — scenario_bayes_factors raises otherwise, so skip.
_sel_node = st.session_state.selected_node
selected_bayes = None
try:
    if _sel_node and _sel_node in evidence:
        selected_bayes = engine.standalone_bayes_factors({_sel_node: evidence[_sel_node]})
    elif _sel_node and _sel_node in soft_evidence:
        selected_bayes = engine.standalone_bayes_factors(
            {}, {_sel_node: soft_evidence[_sel_node]})
except ValueError:
    selected_bayes = None


# ===========================================================================
# HEADER
# ===========================================================================


def _load_eval_badge() -> Optional[str]:
    """Header badge from the committed translator-eval snapshot (T03/D2).

    Returns None if the snapshot is missing (e.g. before `pixi run translator-eval`).
    """
    snap = ROOT / "tests" / "golden" / "translator" / "_eval_snapshot.json"
    try:
        m = json.loads(snap.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    f1, acc = m.get("node_f1"), m.get("state_accuracy_given_node_match")
    f1s = f"{f1:.2f}" if isinstance(f1, (int, float)) else "—"
    accs = f"{acc:.2f}" if isinstance(acc, (int, float)) else "—"
    return (
        "<div class='sb-provider' style='display:inline-block;margin:0 0 0.4rem;'>"
        f"📏 translator eval: n={m.get('n_records', '?')} ({m.get('gate', '?')}) · "
        f"node-F1 {f1s} · state-acc {accs} · "
        f"{m.get('n_nodes_covered', '?')}/{m.get('n_observable_nodes', '?')} nodes</div>"
    )


st.markdown(
    "<div class='demo-title'>Adaptive Scenario Probability Framework — Strait of Hormuz</div>",
    unsafe_allow_html=True,
)
_eval_badge = _load_eval_badge()
if _eval_badge:
    st.markdown(_eval_badge, unsafe_allow_html=True)
with st.expander("How this model works", expanded=False):
    model_explainer.render_overview(st, TOPOLOGY)


# ===========================================================================
# PINNED TOP BAND — scenario cards + probability evolution
# ===========================================================================

scenario_cards.render_scenario_outlook(st, ci_table)

evolution_chart.render_evolution_chart(
    st, st.session_state.observations,
    engine=get_engine(TOPOLOGY), cached_ci=cached_credible_intervals,
    locked_spec_json=_locked_spec_json(), topology=TOPOLOGY,
)


# ===========================================================================
# TOP-LEVEL VIEW NAV — Network & model / Observations / Audit trail / Edges
# ===========================================================================
# Deliberately NOT st.tabs: st.tabs keeps every tab body mounted and merely
# CSS-hides the inactive ones. That breaks the vis.js (streamlit-agraph) DAG
# canvas — when the Network tab is re-shown, vis.js refits against a stale /
# zero-size container and renders zoomed-in or blank. A session-state-driven
# selector with conditional rendering re-mounts only the active view on each
# switch, so the graph always sizes correctly. (agraph exposes no `key`, so
# this is the only way to force the clean remount.)

_VIEW_NET = "🕸️  Network & model"
_VIEW_OBS = "📝  Observations"
_VIEW_TRIAGE = "⚖️  Triage"
_VIEW_AUDIT = "🔎  Audit trail"
_VIEW_EDGES = "🧭  Edge rationale"

active_view = st.segmented_control(
    "View",
    [_VIEW_NET, _VIEW_OBS, _VIEW_TRIAGE, _VIEW_AUDIT, _VIEW_EDGES],
    default=_VIEW_NET,
    key="active_view",
    label_visibility="collapsed",
)
# Single-select segmented_control lets the user deselect the active chip
# (returns None); keep exactly one view active, the way tabs behave.
if not active_view:
    active_view = _VIEW_NET


# ---------------------------------------------------------------------------
# TAB 1 — Network & model (interactive graph + click-to-override / explain)
# ---------------------------------------------------------------------------

if active_view == _VIEW_NET:
    network_view.render(
        st, all_marginals=all_marginals, evidence=evidence,
        soft_evidence=soft_evidence, node_ci_table=node_ci_table,
        observed_day_map=observed_day_map, observed_meta=observed_meta,
        selected_bayes=selected_bayes, topology=TOPOLOGY,
    )

    with st.expander("Appendix — math and implementation details", expanded=False):
        model_explainer.render_appendix(st)


# ---------------------------------------------------------------------------
# TAB 2 — Edge rationale (why each arrow is there, and why some aren't)
# ---------------------------------------------------------------------------

if active_view == _VIEW_EDGES:
    edge_rationale.render(st, TOPOLOGY)
# ---------------------------------------------------------------------------
# TRIAGE — human-in-the-loop review queue (T12, in-session)
# ---------------------------------------------------------------------------

if active_view == _VIEW_TRIAGE:
    triage_view.render(st)

# ---------------------------------------------------------------------------
# TAB 3 — Observations (latest translation + day-grouped log)
# ---------------------------------------------------------------------------

if active_view == _VIEW_OBS:
    observation_log.render(st)
# ---------------------------------------------------------------------------
# TAB 4 — Audit trail (full width, grouped tables)
# ---------------------------------------------------------------------------

if active_view == _VIEW_AUDIT:
    audit_view.render(st, engine=engine, evidence=evidence, soft_evidence=soft_evidence)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.caption(
    "Probabilities are illustrative — CPTs are expert-elicited, not "
    "calibrated from historical data. Translator output is an LLM reading "
    "of each headline and should be reviewed before acting on the resulting "
    "probabilities. See README for the BN-vs-HMM rationale."
)
