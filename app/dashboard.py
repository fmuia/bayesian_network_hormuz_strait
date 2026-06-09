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
import os
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st
from streamlit_agraph import agraph

from src.evidence import EXAMPLE_HEADLINES, Observation
from src.inference import BNInferenceEngine
from src.network import SCENARIO_NARRATIVES, STATES, build_network
from src.sensitivity import (
    node_credible_intervals,
    scenario_credible_intervals,
)
from src.translator import (
    SOURCE_TYPE_CREDIBILITY,
    Article,
    TranslatorError,
    TranslatorResult,
    available_providers,
    fake_forced_by_env,
    is_available as translator_available,
    structured_enabled,
    translate_article,
)
from src.translator_pipeline import run_structured
from src.viz import TOPOLOGY_LAYOUT, build_agraph_payload, render_network_png
from src.elicitation.export import spec_from_dict

# Ensure sibling modules (elicitation_panel) import under both `streamlit run`
# and the test harness.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import elicitation_panel  # noqa: E402
import state  # noqa: E402  (sibling module; relies on the sys.path insert above)
from state import (  # noqa: E402
    build_review_item as _build_review_item,
    current_evidence as _merged_evidence,
    delete_named_session as _delete_named_session,
    inject_review_item as _inject_review_item,
    load_session_store as _load_session_store,
    record_observation as _append_observation,
    remove_review_item as _remove_from_review,
    restore_named_session as _restore_named_session,
    save_named_session as _save_named_session,
)

# ---------------------------------------------------------------------------
# Page setup & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Strait of Hormuz — Adaptive Scenario Probabilities",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette + scenario labels live in app/theme.py (Plan 5 P3) so the chart
# components can share them without importing the dashboard.
from theme import (  # noqa: E402
    AMBER, GREEN, MUTED, NAVY, PANEL, RED, ROOT_DRIVER_STYLE, RULE,
    SCENARIO_COLOR, SCENARIO_LABEL, SCENARIO_KEYS, TEAL,
)

from components import evolution_chart, scenario_cards  # noqa: E402

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


def _render_model_overview(topology: str = TOPOLOGY) -> None:
    if topology == "latent_regime":
        scenario_clause = (
            "a latent <b>Scenario</b> regime that <i>generates</i> the damage, "
            "duration, and diplomatic-path outcomes (with context parents US "
            "military response and strait closure)"
        )
    else:
        scenario_clause = (
            "a terminal <b>Scenario</b> node classified from the damage, "
            "duration, and diplomatic-path outcomes"
        )
    st.markdown(
        "<div class='explain'>"
        "<p>The Bayesian network encodes qualitative causal structure "
        "between four <b>root drivers</b> (negotiations, regime "
        "stability, third-party mediation, sanctions trajectory), "
        "<b>eight intermediate nodes</b> (Iran-aligned militia attacks, tanker "
        "incidents, US military response, strait closure, energy "
        "infrastructure damage, conflict duration, diplomatic path, "
        f"oil price regime), and {scenario_clause} "
        "whose three states correspond to the client's strategic "
        "scenarios.</p>"
        "<h4>Two layers</h4>"
        "<p>A free-text headline is passed through an LLM translator "
        "that extracts BN-relevant probabilistic assignments (e.g. "
        "<i>\"Fourth tanker incident in two weeks\"</i> gives a high "
        "probability to <code>Tanker_Incidents = frequent</code>). "
        "Those soft assignments become BN evidence; variable-elimination "
        "propagates them and yields the posterior distribution at "
        "every node.</p>"
        "<h4>Scenario definitions</h4>"
        "<ul>"
        f"<li><b style='color:{GREEN};'>Stress Mitigates</b> — "
        f"{SCENARIO_NARRATIVES['Stress_Mitigates']}</li>"
        f"<li><b style='color:{AMBER};'>Prolonged Conflict</b> — "
        f"{SCENARIO_NARRATIVES['Prolonged_Conflict']}</li>"
        f"<li><b style='color:{RED};'>Severe Closure</b> — "
        f"{SCENARIO_NARRATIVES['Severe_Closure']}</li>"
        "</ul>"
        "<h4>Reading the graph</h4>"
        "<p>Teal-filled nodes are the ones for which evidence has "
        "been set (whether by the translator or a manual override). "
        "Unobserved nodes display the most likely state under the "
        "current posterior. Root drivers use distinct color families "
        "so they are easy to distinguish visually.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_model_appendix() -> None:
    st.markdown(
        r"""
        ### Appendix: implementation details

        The model is a discrete Bayesian network with posterior updates via exact variable elimination.

        **Inference rule**

        $$
        P(S \mid E=e) =
        \frac{\sum_{z} P(S, z, e)}{\sum_{s}\sum_{z} P(s, z, e)}
        $$

        where $S$ is the scenario node and $z$ are latent/unobserved nodes.

        **Translator-to-evidence pipeline**

        1. A headline is parsed into a set of node assignments constrained to valid node states.
        2. Each assignment is appended as an observation with day and source metadata.
        3. Latest assignment wins on node conflicts when merged into current evidence.
        4. Inference is re-run and scenario cards + node marginals are refreshed.

        **Uncertainty panel**

        Credible intervals are estimated by resampling every CPT column from a Dirichlet distribution centred on the elicited point estimate and rerunning inference:

        $$
        \theta_{j,\cdot}^{(m)} \sim \text{Dirichlet}(\alpha_{j,\cdot}),
        \qquad \alpha_{j,\cdot} = \kappa \cdot \theta_{j,\cdot}^{\text{point}}
        $$

        with concentration $\kappa = 20$ and $m = 200$ draws. Each draw perturbs **all** CPTs jointly and the full network is re-run under the current evidence, so the resulting posterior samples reflect *global* parameter uncertainty, not a per-node local variation.

        The 10th–90th percentiles across samples give an 80% credible interval per node per state, exposed in two places:

        - **Scenario cards** (top band): headline CIs for the three scenarios.
        - **Node detail panel** (right of the Network tab): per-node dumbbells with a robustness badge (🟢 robust < ±8 pp · 🟡 moderate ±8–20 pp · 🔴 fragile > ±20 pp). Hard-observed nodes collapse to deltas; soft-observed nodes keep their CIs because the posterior still varies under CPT resampling.
        """
    )


# ---------------------------------------------------------------------------
# Translator stream (compact single-line)
# ---------------------------------------------------------------------------

STAGE_ICON = {
    "init": "🔌",
    "thinking": "💭",
    "response": "✍️",
    "parsing": "🧩",
    "validated": "✅",
}
STAGE_LABEL = {
    "init": "Connecting to model",
    "thinking": "Thinking",
    "response": "Receiving response",
    "parsing": "Parsing assignments",
    "validated": "Validated",
}


# Sidebar source-type options -> (Article.source_type, explicit credibility).
# "(unspecified)" = analyst paste at full trust (w=1.0); the rest defer to the
# per-source-type table (credibility=None -> looked up in translate_article).
_FULL_TRUST_LABEL = "(unspecified — full trust)"
_SOURCE_TYPE_OPTIONS = [_FULL_TRUST_LABEL] + list(SOURCE_TYPE_CREDIBILITY.keys())


def _resolve_source(label: str):
    """Map a sidebar source-type label to (source_type, credibility-or-None)."""
    if label == _FULL_TRUST_LABEL:
        return "unknown", 1.0
    return label, None  # None -> translate_article looks up the table


def _run_translator(article_fields: dict, stream_slot, *, provider: Optional[str] = None) -> None:
    def _write(kind: str, stage: str, detail: str) -> None:
        icon = STAGE_ICON.get(stage, "•")
        label = STAGE_LABEL.get(stage, stage.capitalize())
        clean = " ".join(detail.split())
        if len(clean) > 120:
            clean = clean[:117] + "…"
        cls = {"live": "stream-line", "done": "stream-line stream-done",
               "err":  "stream-line stream-error"}[kind]
        stream_slot.markdown(
            f"<div class='{cls}'>{icon} <b>{label}</b> — {clean}</div>",
            unsafe_allow_html=True,
        )

    _write("live", "init", "starting model call…")

    def on_step(stage: str, detail: str) -> None:
        _write("live", stage, detail)

    source_type, credibility = _resolve_source(
        article_fields.get("source_type_label", _FULL_TRUST_LABEL)
    )
    article = Article(
        headline=article_fields["headline"],
        body=article_fields.get("body", ""),
        source=article_fields.get("source", ""),
        source_type=source_type,
    )
    # T06e: when the structured toggle is on, the structured pipeline (extract →
    # map → aggregate) PRODUCES the injected assignments; otherwise the single-
    # call path does. Structured costs 2 LLM calls and derives relevance as
    # yes/no (no "partial"); the single-call path keeps the richer relevance.
    use_structured = st.session_state.get("use_structured")
    claims = mappings = None
    try:
        if use_structured:
            result, claims, mappings = run_structured(
                article, credibility=credibility, provider=provider, on_step=on_step
            )
        else:
            result = translate_article(
                article, credibility=credibility, provider=provider, on_step=on_step
            )
    except TranslatorError as exc:
        raw = getattr(exc, "raw_response", "")
        _write("err", "validated", f"failed: {exc}")
        st.session_state.translator_error = str(exc)
        st.session_state.translator_raw = raw
        st.session_state.last_translation = None
        return

    st.session_state.translator_error = None
    st.session_state.translator_raw = result.raw_response
    st.session_state.last_translation = {
        "headline": result.headline,
        "assignments": [asdict(a) for a in result.assignments],
        "rationale": result.rationale,
        "model": result.model,
        "provider": result.provider,
        "relevance": result.relevance,
    }
    if use_structured:
        st.session_state.last_translation["claims"] = [asdict(c) for c in claims]
        st.session_state.last_translation["claim_mappings"] = [asdict(m) for m in mappings]
        st.session_state.last_translation["structured_assignments"] = [
            asdict(a) for a in result.assignments
        ]

    # B3: an off-topic article abstains — logged, but no evidence injected.
    if result.relevance == "no":
        _write("done", "validated", "not relevant — no evidence injected")
        return

    if result.assignments:
        # T12: route to HITL review when flagged (partial) or when the analyst
        # has turned on "review before inject"; otherwise auto-approve (inject).
        needs_review = (
            result.relevance == "partial"
            or st.session_state.get("review_before_inject", False)
        )
        st.session_state.last_translation["pending_review"] = needs_review
        item = _build_review_item(result)
        if needs_review:
            st.session_state.review_queue.append(item)
            _write(
                "done", "validated",
                f"{len(result.assignments)} assignment(s) → queued for review "
                f"(not yet injected) · see the Triage view",
            )
        else:
            _inject_review_item(item)
            _write(
                "done", "validated",
                f"{len(result.assignments)} assignment(s) · auto-approved · model {result.model}",
            )
    else:
        st.session_state.translator_error = (
            "Translator returned no assignments — the headline does not map "
            "to any node in this BN schema. Try a Strait-of-Hormuz-specific "
            "headline or use the Network tab to set a node manually."
        )
        _write("err", "validated", "no assignments produced")


# ===========================================================================
# SIDEBAR
# ===========================================================================

providers = available_providers()
provider_labels = {"claude-code": "Claude Code", "openai": "OpenAI API"}

# Offline `fake` translator (deterministic fixtures, no network). Default the
# dev toggle on when TRANSLATOR_PROVIDER=fake forces it, or when no real backend
# is available (so the app is always playable). The toggle widget owns the state.
if "use_fake_translator" not in st.session_state:
    st.session_state.use_fake_translator = (
        fake_forced_by_env() or not translator_available()
    )
if "use_structured" not in st.session_state:
    st.session_state.use_structured = structured_enabled()

with st.sidebar:
    st.markdown(
        "<div class='sb-header'>"
        "<div class='sb-header-title'>Scenario Session Controls</div>"
        "<div class='sb-header-sub'>Translator, observations, and state</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # -- Fake-translator dev toggle (offline, deterministic) ---------------
    use_fake = st.toggle(
        "Use fake translator (offline)",
        key="use_fake_translator",
        help="Deterministic fixtures, no network or API key. For dev / manual "
             "verification without spending LLM calls.",
    )
    use_structured = st.toggle(
        "Experimental: structured pipeline",
        key="use_structured",
        help="Span-grounded structured reasoning (B2): extract atomic claims → map "
             "each to a node → aggregate. When on, this PRODUCES the injected "
             "assignments (every one cites verbatim spans) and resists prompt "
             "injection. Costs 2 LLM calls and derives relevance as yes/no only. "
             "Off = the single-call path (1 call, richer relevance).",
    )
    review_before_inject = st.toggle(
        "Require review before inject",
        key="review_before_inject",
        help="Human-in-the-loop: hold every translation in the Triage view for "
             "approve / edit / reject before it affects the model. Partial-relevance "
             "translations are always held regardless of this toggle.",
    )
    translator_on = use_fake or translator_available()

    # -- Provider chip (one line) ------------------------------------------
    if use_fake:
        st.markdown(
            "<div class='sb-provider'>● Translator: fake (offline dev)</div>",
            unsafe_allow_html=True,
        )
    elif translator_on:
        primary = provider_labels[providers[0]]
        st.markdown(
            f"<div class='sb-provider'>● Translator: {primary}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='sb-provider warn'>⚠ No translator backend</div>",
            unsafe_allow_html=True,
        )

    _n_review = len(st.session_state.review_queue)
    if _n_review:
        st.markdown(
            f"<div class='sb-provider warn'>⏳ {_n_review} awaiting review — see Triage</div>",
            unsafe_allow_html=True,
        )

    # -- Day row ----------------------------------------------------------
    todays_count = sum(
        1 for o in st.session_state.observations
        if o["day"] == st.session_state.current_day
    )
    day_l, day_r = st.columns([1, 1], gap="small")
    with day_l:
        st.markdown(
            f"<div class='day-pill'>DAY {st.session_state.current_day}</div>"
            f"<div class='sb-hint'>• {todays_count} obs today</div>",
            unsafe_allow_html=True,
        )
    with day_r:
        if st.button("▶ Advance", width="stretch", type="secondary", key="adv_day"):
            st.session_state.current_day += 1
            st.rerun()

    st.markdown("<div class='sb-title'>Translate a headline or article</div>",
                unsafe_allow_html=True)

    with st.form("headline_form", clear_on_submit=True):
        headline_input = st.text_area(
            "News headline",
            placeholder="e.g. 'Iran suspends Hormuz traffic inspections'",
            height=72,
            disabled=not translator_on,
            label_visibility="collapsed",
        )
        with st.expander("Add article body & source (optional)"):
            body_input = st.text_area(
                "Article body",
                placeholder="Paste the article body — qualifiers in the body "
                            "(e.g. 'third such incident this week', 'no injuries') "
                            "disambiguate states the headline alone can't.",
                height=120,
                disabled=not translator_on,
            )
            source_input = st.text_input(
                "Source (outlet or domain)", placeholder="e.g. Reuters",
                disabled=not translator_on,
            )
            source_type_input = st.selectbox(
                "Source type (sets credibility weight w)",
                _SOURCE_TYPE_OPTIONS, index=0, disabled=not translator_on,
            )
        submitted = st.form_submit_button(
            "Translate & observe", type="primary",
            disabled=not translator_on, width="stretch",
        )
        if submitted and headline_input.strip():
            st.session_state.pending_article = {
                "headline": headline_input.strip(),
                "body": body_input.strip(),
                "source": source_input.strip(),
                "source_type_label": source_type_input,
            }

    # Stream slot lives just below the form — compact, single-line.
    stream_slot = st.empty()

    # Run translator *after* the slot is in the sidebar, so updates appear here.
    if st.session_state.pending_article is not None:
        article_fields = st.session_state.pending_article
        st.session_state.pending_article = None
        _run_translator(
            article_fields, stream_slot, provider="fake" if use_fake else None
        )

    with st.expander("Examples", expanded=False):
        for idx, ex in enumerate(EXAMPLE_HEADLINES):
            if st.button(
                ex.text, key=f"ex_{idx}",
                width="stretch", disabled=not translator_on,
            ):
                st.session_state.pending_article = {
                    "headline": ex.text, "body": "", "source": "",
                    "source_type_label": _FULL_TRUST_LABEL,
                }
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sb-title'>Named sessions</div>", unsafe_allow_html=True)
    saved_sessions = _load_session_store()
    session_name = st.text_input(
        "Session name",
        placeholder="e.g. baseline-briefing",
        key="session_name_input",
        label_visibility="collapsed",
    ).strip()

    sess_cols = st.columns([1, 1], gap="small")
    with sess_cols[0]:
        if st.button("Save session", width="stretch", key="save_named_session"):
            if not session_name:
                st.warning("Enter a session name before saving.")
            else:
                _save_named_session(session_name)
                st.success(f"Saved session '{session_name}'.")
                st.rerun()
    with sess_cols[1]:
        load_name = st.selectbox(
            "Load named session",
            options=[""] + sorted(saved_sessions.keys()),
            key="load_named_session_select",
            label_visibility="collapsed",
        )
        if st.button("Load", width="stretch", key="load_named_session"):
            if not load_name:
                st.warning("Choose a saved session to load.")
            elif _restore_named_session(load_name):
                st.success(f"Loaded session '{load_name}'.")
                st.rerun()
            else:
                st.error("Could not load that saved session.")

    if saved_sessions:
        delete_name = st.selectbox(
            "Delete named session",
            options=[""] + sorted(saved_sessions.keys()),
            key="delete_named_session_select",
            label_visibility="collapsed",
        )
        if st.button("Delete selected", width="stretch", key="delete_named_session"):
            if delete_name and _delete_named_session(delete_name):
                st.success(f"Deleted session '{delete_name}'.")
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Reset session", width="stretch", key="reset_all"):
        st.session_state.observations = []
        st.session_state.current_day = 1
        st.session_state.last_translation = None
        st.session_state.translator_error = None
        st.session_state.translator_raw = ""
        st.session_state.selected_node = None
        st.rerun()


# ===========================================================================
# MAIN COMPUTATION
# ===========================================================================

# Plan 4 elicitation layer: run/load an elicitation, inspect reasonings & scores,
# override columns, and lock it — the rest of the dashboard then runs on it.
with st.expander(
    "🧪 Elicitation layer — run / load / override / lock the CPTs", expanded=False
):
    elicitation_panel.render(st, topology=TOPOLOGY)

engine = get_engine(TOPOLOGY)
engine.clear_evidence()
evidence, soft_evidence = _merged_evidence()
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

# Map each observed node to the latest day it was set.
observed_day_map: Dict[str, int] = {}
for obs in st.session_state.observations:
    for node in obs.get("assignments", {}):
        observed_day_map[node] = obs["day"]
    for node in obs.get("soft_assignments", {}):
        observed_day_map[node] = obs["day"]


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
    _render_model_overview(TOPOLOGY)


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
_VIEW_TRIAGE = "🧪  Triage"
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
    net_col, detail_col = st.columns([2.35, 1.0], gap="large")

    with net_col:
        with st.container(border=True):
            root_chip_html = "".join(
                (
                    f"<span class='root-chip' style='background:{bg};"
                    f"border:1px solid {border}; color:{border};'>"
                    f"{node.replace('_', ' ')}</span>"
                )
                for node, (bg, border) in ROOT_DRIVER_STYLE.items()
            )
            st.markdown(
                "<div class='card-title'>Interactive DAG — click a node</div>"
                "<div class='card-sub'>Fixed layout for at-a-glance reading. "
                "Hover nodes to see full percentages. Root drivers use dedicated "
                "color families for clearer separation. Node labels show posterior "
                "output after propagating all injected evidence.</div>"
                f"<div>{root_chip_html}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:0.25rem;'></div>", unsafe_allow_html=True)
            _topo_edges, _topo_levels = TOPOLOGY_LAYOUT[TOPOLOGY]
            nodes, edges, config = build_agraph_payload(
                all_marginals,
                observed=evidence,
                observed_day=observed_day_map,
                edges=_topo_edges,
                node_level=_topo_levels,
            )
            clicked = agraph(nodes=nodes, edges=edges, config=config)
            st.markdown("<div style='height:0.2rem;'></div>", unsafe_allow_html=True)
            if clicked and clicked in STATES:
                if st.session_state.selected_node != clicked:
                    st.session_state.selected_node = clicked
                    st.rerun()

    with detail_col:
        sel = st.session_state.selected_node
        with st.container(border=True):
            st.markdown(
                "<div class='card-title'>Posterior</div>",
                unsafe_allow_html=True,
            )
            if sel and sel in STATES:
                marginal = all_marginals[sel]
                sorted_states = list(STATES[sel])
                if sel in evidence:
                    tip_text = (
                        f"Hard evidence: {evidence[sel]} "
                        f"(day {observed_day_map.get(sel, '?')}). "
                        "No residual model uncertainty — the node is "
                        "pinned to the observed state."
                    )
                elif sel in soft_evidence:
                    dist = soft_evidence[sel]
                    top_state = max(dist, key=dist.get)
                    tip_text = (
                        f"Soft evidence from headlines: best-supported state "
                        f"{top_state} (likelihood ratios ε, scaled to 1.0; "
                        f"day {observed_day_map.get(sel, '?')}). "
                        "Intervals show how the posterior shifts when "
                        "CPT parameters are resampled (Dirichlet, "
                        "concentration = 20, m = 200)."
                    )
                else:
                    tip_text = (
                        "Posterior marginal after propagating all injected "
                        "evidence. Intervals come from CPT resampling "
                        "(Dirichlet, concentration = 20, m = 200)."
                    )
                tip_attr = tip_text.replace("'", "&#39;")
                st.markdown(
                    f"<div class='card-sub' style='display:flex; "
                    f"align-items:center; gap:0.35rem;'>"
                    f"<b>{sel.replace('_',' ')}</b>"
                    f"<span title='{tip_attr}' "
                    f"style='cursor:help; color:{MUTED}; "
                    f"font-size:0.9rem; font-weight:500;'>ⓘ</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if sel in evidence:
                    st.altair_chart(
                        _flat_bar_chart(marginal, sorted_states),
                        width="stretch",
                    )
                else:
                    node_ci = node_ci_table[sel]
                    st.markdown(
                        _robustness_badge_html(node_ci, sorted_states),
                        unsafe_allow_html=True,
                    )
                    ci_df = _ci_dataframe(node_ci, sorted_states)
                    st.altair_chart(
                        _dumbbell_chart(ci_df, sorted_states),
                        width="stretch",
                    )
            else:
                st.markdown(
                    "<div class='card-sub'>Click a node in the graph to inspect "
                    "its posterior distribution.</div>",
                    unsafe_allow_html=True,
                )

        with st.container(border=True):
            st.markdown(
                "<div class='card-title' style='margin-bottom:0.2rem;'>"
                "Override</div>",
                unsafe_allow_html=True,
            )
            if sel and sel in STATES:
                states = list(STATES[sel])
                n = len(states)
                default_pct = 100 // n
                remainder = 100 - default_pct * n
                vals: Dict[str, int] = {}
                for i, state in enumerate(states):
                    key = f"soft_{sel}_{state}"
                    init = default_pct + (remainder if i == 0 else 0)
                    vals[state] = st.slider(
                        state, 0, 100,
                        value=st.session_state.get(key, init),
                        step=1, key=key,
                    )
                total = sum(vals.values())
                colour = GREEN if total == 100 else RED
                st.markdown(
                    f"<div style='font-size:0.85rem; margin-top:0.35rem;'>"
                    f"Total: <b style='color:{colour}'>{total}%</b></div>",
                    unsafe_allow_html=True,
                )
                note = st.text_input(
                    "Note (optional)", key=f"note_{sel}",
                    placeholder="What drove this override?",
                )
                if st.button(
                    "Set observation",
                    type="primary",
                    disabled=(total != 100),
                    key=f"set_{sel}",
                ):
                    pretty = ", ".join(
                        f"{s} {v}%" for s, v in vals.items() if v > 0
                    )
                    # Collapse to a hard assignment when a single state is 100%.
                    if max(vals.values()) == 100:
                        pinned = next(s for s, v in vals.items() if v == 100)
                        _append_observation(
                            headline=note.strip() or f"Manual: {sel} = {pinned}",
                            assignments={sel: pinned},
                            rationale="Set directly by the analyst via the network.",
                            per_assignment_reasons={sel: "Manual override."},
                            source="manual",
                        )
                    else:
                        dist = {s: v / 100.0 for s, v in vals.items()}
                        _append_observation(
                            headline=note.strip()
                                or f"Manual soft: {sel} ({pretty})",
                            assignments={},
                            soft_assignments={sel: dist},
                            rationale="Soft override set directly by the analyst.",
                            per_assignment_reasons={sel: f"Manual soft override: {pretty}."},
                            source="manual",
                        )
                    st.rerun()
            else:
                st.markdown(
                    "<div class='card-sub'>Select a node first to enable manual "
                    "override controls.</div>",
                    unsafe_allow_html=True,
                )

    with st.expander("Appendix — math and implementation details", expanded=False):
        _render_model_appendix()


# ---------------------------------------------------------------------------
# TAB 2 — Edge rationale (why each arrow is there, and why some aren't)
# ---------------------------------------------------------------------------

_EDGE_RATIONALE: list[tuple[str, str, str]] = [
    ("Iranian_Regime_Stability", "Iran_Aligned_Militia_Attacks",
     "An unstable or pressured regime has stronger incentive to lash out "
     "externally via its militia network; a stable regime can afford "
     "restraint and tighter command over proxies."),
    ("Sanctions_Trajectory", "Iran_Aligned_Militia_Attacks",
     "Tightening sanctions remove peaceful off-ramps and push Tehran to "
     "impose cost asymmetrically through proxies; easing sanctions "
     "reward restraint."),
    ("Iran_Aligned_Militia_Attacks", "Tanker_Incidents",
     "Houthi drone/missile strikes and IRGC-linked harassment are the "
     "direct mechanism behind most attacks on Gulf and Red Sea shipping."),
    ("US_Iran_Negotiations", "Tanker_Incidents",
     "Active back-channels dampen incidents via restraint signals; "
     "negotiation breakdowns remove that brake."),
    ("Tanker_Incidents", "US_Military_Response",
     "Observable kinetic events on shipping (strikes, seizures) are the "
     "visible trigger for carrier redeployments, escorts, and retaliatory "
     "strikes."),
    ("Sanctions_Trajectory", "US_Military_Response",
     "A hawkish sanctions posture correlates with political willingness "
     "to use force; an easing posture correlates with restraint."),
    ("Tanker_Incidents", "Strait_Operationally_Closed",
     "Insurance premiums, charterer avoidance, and convoy logistics mean "
     "enough incidents produce de facto closure even without Iranian "
     "mining."),
    ("US_Military_Response", "Strait_Operationally_Closed",
     "A major US response can either re-open traffic via escorts or "
     "provoke Iranian mining/closure attempts — both captured as "
     "dependence."),
    ("US_Military_Response", "Energy_Infrastructure_Damage",
     "Strikes on IRGC naval assets or Iranian oil facilities — and "
     "Iranian retaliation on Saudi/UAE/US infrastructure — are driven by "
     "the intensity of the military response."),
    ("Strait_Operationally_Closed", "Energy_Infrastructure_Damage",
     "Extended closure correlates with the broader escalation regime "
     "that brings production and export infrastructure into the "
     "crosshairs."),
    ("US_Iran_Negotiations", "Conflict_Duration",
     "Active direct talks accelerate resolution; breakdowns prolong the "
     "crisis by removing the obvious exit."),
    ("Third_Party_Mediation", "Conflict_Duration",
     "Qatari, Omani, or Chinese mediation compresses timelines to a "
     "deal by providing face-saving channels."),
    ("US_Military_Response", "Conflict_Duration",
     "A major response commits the US to a protracted campaign; no "
     "response allows quicker de-escalation."),
    ("US_Iran_Negotiations", "Diplomatic_Resolution_Path",
     "The status of direct bilateral talks is the primary determinant "
     "of whether a resolution path stays open."),
    ("Third_Party_Mediation", "Diplomatic_Resolution_Path",
     "External facilitators open or preserve channels precisely when "
     "direct talks stall."),
    ("Iranian_Regime_Stability", "Diplomatic_Resolution_Path",
     "A stable regime can credibly commit and bind hardliner factions; "
     "an unstable regime cannot deliver on any deal it signs."),
    ("Strait_Operationally_Closed", "Oil_Price_Regime",
     "Roughly 20% of seaborne oil transits Hormuz — closure is the "
     "single largest supply-shock lever in the model."),
    ("Energy_Infrastructure_Damage", "Oil_Price_Regime",
     "Damaged production or export capacity removes barrels from the "
     "market for quarters, not days."),
    ("Energy_Infrastructure_Damage", "Scenario",
     "Severe infrastructure damage pushes the scenario toward "
     "Prolonged_Conflict or Severe_Closure."),
    ("Conflict_Duration", "Scenario",
     "A long conflict precludes Stress_Mitigates and favors the two "
     "heavier scenarios."),
    ("Diplomatic_Resolution_Path", "Scenario",
     "An open resolution path is the main driver of the "
     "Stress_Mitigates outcome."),
]

_EDGE_OMISSIONS: list[tuple[str, str, str]] = [
    ("Iran_Aligned_Militia_Attacks", "US_Military_Response",
     "Assumed mediated by Tanker_Incidents: Washington reacts to "
     "observable kinetic events, not militia posture in the abstract. "
     "Known limitation — US base attacks in Iraq/Syria have historically "
     "triggered direct US strikes without any tanker incident."),
    ("Iranian_Regime_Stability", "Tanker_Incidents",
     "Mediated by Iran_Aligned_Militia_Attacks: regime posture reaches "
     "the water only through the proxy network."),
    ("Sanctions_Trajectory", "Tanker_Incidents",
     "Mediated by Iran_Aligned_Militia_Attacks — same reasoning. The "
     "direct sanctions→incidents arrow is absorbed into the militia "
     "channel."),
    ("US_Military_Response", "Oil_Price_Regime",
     "Mediated by Strait_Operationally_Closed and "
     "Energy_Infrastructure_Damage. The oil market prices physical "
     "flows and capacity, not force posture directly."),
    ("Strait_Operationally_Closed", "Scenario",
     "Mediated by Oil_Price_Regime and Energy_Infrastructure_Damage. "
     "The Scenario node classifies on downstream outcomes, not on the "
     "mechanism that produced them."),
    ("Third_Party_Mediation", "Iran_Aligned_Militia_Attacks",
     "Mediation is modeled as operating on negotiations and duration, "
     "not on kinetic tempo. Debatable — Oman has historically passed "
     "de-escalation messages to the IRGC during flashpoints."),
    ("Iranian_Regime_Stability", "US_Military_Response",
     "Omitted for parsimony: US response is driven by events plus "
     "sanctions posture in the model, not by Iranian internal politics "
     "directly."),
]

# --- Latent-regime topology (Plan 1) -----------------------------------------
# The three {D,T,P} -> Scenario classifier edges reverse into emissions, and
# Scenario gains the context parents {US_Military_Response, Strait_Operationally_Closed}.
_LATENT_SCENARIO_EDGES: list[tuple[str, str, str]] = [
    ("US_Military_Response", "Scenario",
     "Context parent of the regime: a major US military response raises the prior "
     "for the Severe/Prolonged regimes before any outcome is observed."),
    ("Strait_Operationally_Closed", "Scenario",
     "Context parent of the regime: observed strait closure is a strong prior signal "
     "for the Severe regime. Retained alongside military response (partly redundant) "
     "for closure-evidence sensitivity — a deliberate parsimony exception."),
    ("Scenario", "Energy_Infrastructure_Damage",
     "Emission: the regime generates damage. The Severe regime concentrates mass on "
     "severe damage but keeps nonzero off-mode mass (overlap is by design)."),
    ("Scenario", "Conflict_Duration",
     "Emission: the regime generates conflict duration (Prolonged/Severe lean long)."),
    ("Scenario", "Diplomatic_Resolution_Path",
     "Emission: the regime generates the diplomatic path (Stress leans open, "
     "Severe leans closed)."),
]
_EDGE_RATIONALE_LATENT: list[tuple[str, str, str]] = (
    [e for e in _EDGE_RATIONALE if e[1] != "Scenario"] + _LATENT_SCENARIO_EDGES
)
_EDGE_OMISSIONS_LATENT: list[tuple[str, str, str]] = (
    [e for e in _EDGE_OMISSIONS if e[1] != "Scenario"] + [
        ("Third_Party_Mediation", "Scenario",
         "Documented blind spot (Plan 1 §A.4): mediation has no direct path to the "
         "regime in v1 — it reaches Scenario only indirectly, through the diplomatic-"
         "path and duration emissions. Accepted for v1; candidate direct parent in "
         "Plan 4."),
    ]
)

_RATIONALE_BY_TOPOLOGY = {
    "labelling": (_EDGE_RATIONALE, _EDGE_OMISSIONS),
    "latent_regime": (_EDGE_RATIONALE_LATENT, _EDGE_OMISSIONS_LATENT),
}


def _fmt_node(name: str) -> str:
    return name.replace("Iran_Aligned", "Iran-Aligned").replace("_", " ")


if active_view == _VIEW_EDGES:
    st.markdown(
        "<div class='card-title'>Why each arrow is (or isn't) in the "
        "network</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Short rationales for every edge in the DAG, plus a list of "
        "plausible connections that were deliberately left out and why. "
        "These are modeling judgments, not settled truths — treat them "
        "as the assumptions behind the current CPTs."
    )

    _rationale, _omissions = _RATIONALE_BY_TOPOLOGY[TOPOLOGY]

    st.markdown("#### Edges present in the model")
    for parent, child, reason in _rationale:
        st.markdown(
            f"**{_fmt_node(parent)}** → **{_fmt_node(child)}**  \n"
            f"<span style='color:#475569'>{reason}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Notable omitted edges")
    st.caption(
        "Connections a domain reader might expect but that are not in "
        "the DAG — either because they're mediated by another node or "
        "because they were dropped for parsimony."
    )
    for parent, child, reason in _omissions:
        st.markdown(
            f"**{_fmt_node(parent)}** ⇢ **{_fmt_node(child)}** "
            f"<span style='color:#b45309'>(omitted)</span>  \n"
            f"<span style='color:#475569'>{reason}</span>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# TRIAGE — human-in-the-loop review queue (T12, in-session)
# ---------------------------------------------------------------------------

if active_view == _VIEW_TRIAGE:
    st.markdown(
        "<div class='card-title'>Triage — translations awaiting review</div>"
        "<div class='card-sub'>Flagged translations (partial relevance, or all of "
        "them when “Require review before inject” is on) wait here and do "
        "<b>not</b> affect the model until you act. Approve, edit a state, or "
        "reject.</div>",
        unsafe_allow_html=True,
    )
    _queue = st.session_state.review_queue
    if not _queue:
        st.info(
            "Nothing awaiting review. Clearly-relevant translations auto-inject; "
            "off-topic ones abstain. Turn on “Require review before inject” in the "
            "sidebar to route everything here first."
        )
    for _item in list(_queue):
        with st.container(border=True):
            _rel = _item.get("relevance", "yes")
            _badge = (
                " <span class='assign-chip' style='background:#FEF3C7;color:#92400E;'>"
                "⚠ partial</span>" if _rel == "partial" else ""
            )
            st.markdown(
                f"<div class='translator-headline'>“{_item['headline']}”{_badge}</div>"
                f"<div class='meta'>day {_item['day']} · {_item['provider']} · "
                f"{_item['model']}</div>",
                unsafe_allow_html=True,
            )
            chips = "".join(
                f"<span class='assign-chip'>{a['node'].replace('_',' ')} = {a['state']}</span>"
                for a in _item["assignments"]
            )
            st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)

            a_col, r_col = st.columns(2, gap="small")
            if a_col.button("✓ Approve", key=f"appr_{_item['id']}", width="stretch",
                            type="primary"):
                _inject_review_item(_item)
                _remove_from_review(_item["id"])
                st.rerun()
            if r_col.button("✕ Reject", key=f"rej_{_item['id']}", width="stretch"):
                _remove_from_review(_item["id"])
                st.rerun()

            with st.expander("Edit states before approving"):
                _overrides = {}
                for a in _item["assignments"]:
                    node = a["node"]
                    _overrides[node] = st.selectbox(
                        node.replace("_", " "),
                        STATES[node],
                        index=STATES[node].index(a["state"]),
                        key=f"edit_{_item['id']}_{node}",
                    )
                if st.button("✓ Approve with edits", key=f"appredit_{_item['id']}",
                             width="stretch"):
                    _inject_review_item(_item, state_overrides=_overrides)
                    _remove_from_review(_item["id"])
                    st.rerun()


# ---------------------------------------------------------------------------
# TAB 3 — Observations (latest translation + day-grouped log)
# ---------------------------------------------------------------------------

if active_view == _VIEW_OBS:
    trans_col, log_col = st.columns([1.0, 1.1], gap="large")

    with trans_col:
        st.markdown("<div class='card-title'>Latest translation</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div class='card-sub'>Translator percentages are evidence inputs "
            "(soft evidence), not posterior outputs.</div>",
            unsafe_allow_html=True,
        )
        if st.session_state.translator_error:
            st.error(st.session_state.translator_error)
        elif st.session_state.last_translation is None:
            st.markdown(
                "<div class='card-sub'>No headline translated yet this "
                "session.</div>",
                unsafe_allow_html=True,
            )
        else:
            t = st.session_state.last_translation
            chips_html = "".join(
                f"<span class='assign-chip'>"
                f"{a['node'].replace('_',' ')} = {a['state']}</span>"
                for a in t["assignments"]
            ) or "<span style='color:#9CA3AF;'>No assignments</span>"
            # B3 relevance badge.
            _rel = t.get("relevance", "yes")
            _rel_badge = {
                "no": "<span class='assign-chip' style='background:#FEE2E2;color:#991B1B;'>"
                      "⛔ not relevant — not injected</span>",
                "partial": "<span class='assign-chip' style='background:#FEF3C7;color:#92400E;'>"
                           "⚠ partial relevance — review before relying on it</span>",
            }.get(_rel, "")
            st.markdown(
                f"""
                <div class='translator-headline'>“{t['headline']}”</div>
                <div class='translator-rationale'>{t['rationale']}</div>
                <div>{_rel_badge}{chips_html}</div>
                <div class='meta'>provider: {t.get('provider','?')} ·
                model: {t['model']} · relevance: {_rel}</div>
                """,
                unsafe_allow_html=True,
            )
            if t.get("pending_review"):
                st.warning(
                    "⏳ **Pending review — not yet injected.** Approve, edit a state, "
                    "or reject it in the **🧪 Triage** view; until then it does not "
                    "affect the model."
                )
            if "claims" in t:
                _claims = t["claims"]
                _maps = t.get("claim_mappings", [])
                _by_span = {m["supporting_span"]: m for m in _maps if m.get("supporting_span")}
                with st.expander(
                    f"Structured pipeline (experimental) — {len(_claims)} claim(s), "
                    f"{len(_maps)} mapped",
                    expanded=False,
                ):
                    st.caption(
                        "Span-grounded atomic claims (B2 step 1) mapped to BN nodes "
                        "(step 2) then aggregated (step 3). Each claim cites a verbatim "
                        "span copied from the article; ungrounded claims are dropped. "
                        "The aggregated output below **is** what was injected."
                    )
                    if t.get("claims_error"):
                        st.warning(f"Structured pipeline failed: {t['claims_error']}")
                    for c in _claims:
                        span = c["verbatim_span"]
                        m = _by_span.get(span)
                        if m:
                            mapped = (
                                f" → **{m['node'].replace('_',' ')} = {m['state']}**"
                            )
                        else:
                            mapped = " → <span style='color:#9CA3AF;'>(no node)</span>"
                        st.markdown(f"- “{span}”{mapped}", unsafe_allow_html=True)
                    if not _claims and not t.get("claims_error"):
                        st.markdown("_No grounded claims extracted._")
                    _agg = t.get("structured_assignments")
                    if _agg is not None:
                        st.markdown("**Aggregated pipeline output (injected):**")
                        if _agg:
                            for a in _agg:
                                st.markdown(
                                    f"- {a['node'].replace('_',' ')} = **{a['state']}**"
                                )
                        else:
                            st.markdown("_No nodes mapped — abstained._")
            if t["assignments"]:
                with st.expander("Per-assignment likelihood ratios (translator soft evidence)"):
                    st.caption(
                        "ε = relative likelihood of the article given each state "
                        "(best-supported state pinned to 1.0); injected as soft "
                        "evidence, not a probability distribution."
                    )
                    for a in t["assignments"]:
                        probs = a.get("state_probs", {})
                        eps_text = " · ".join(
                            (f"**{k.replace('_',' ')}: {float(v):.2f}**"
                             if abs(float(v) - 1.0) < 1e-6
                             else f"{k.replace('_',' ')}: {float(v):.2f}")
                            for k, v in probs.items()
                        )
                        eps_suffix = f"  \n  ε: {eps_text}" if eps_text else ""
                        st.markdown(
                            f"- **{a['node'].replace('_',' ')} = "
                            f"`{a['state']}`** — {a['reason']}"
                            f"{eps_suffix}"
                        )
        if st.session_state.translator_raw:
            with st.expander("Raw model response (debug)"):
                st.code(st.session_state.translator_raw, language="json")

    with log_col:
        st.markdown("<div class='card-title'>Observation log (injected evidence inputs)</div>",
                    unsafe_allow_html=True)
        if not st.session_state.observations:
            st.markdown(
                "<div class='card-sub'>Translate a headline (or override a "
                "node in the Network tab) to begin.</div>",
                unsafe_allow_html=True,
            )
        else:
            grouped: Dict[int, List[Dict]] = {}
            for obs in st.session_state.observations:
                grouped.setdefault(obs["day"], []).append(obs)
            for day in sorted(grouped, reverse=True):
                day_obs = grouped[day]
                st.markdown(
                    f"<div class='day-block-header'>Day {day} · "
                    f"{len(day_obs)} observation(s)</div>",
                    unsafe_allow_html=True,
                )
                for idx, obs in enumerate(day_obs):
                    hard_assign_str = " · ".join(
                        f"{n.replace('_',' ')} = {s}"
                        for n, s in obs.get("assignments", {}).items()
                    )
                    soft_assign_str = " · ".join(
                        (
                            f"{node.replace('_',' ')} ≈ {max(dist, key=dist.get)} "
                            f"({max(dist.values())*100:0.1f}%, soft)"
                        )
                        for node, dist in obs.get("soft_assignments", {}).items()
                    )
                    assign_str = " · ".join(
                        part for part in [hard_assign_str, soft_assign_str] if part
                    )
                    first_cls = " obs-row-first" if idx == 0 else ""
                    row_col, btn_col = st.columns([20, 1])
                    with row_col:
                        st.markdown(
                            f"<div class='obs-row{first_cls}'>"
                            f"<div class='obs-headline'>{obs['headline']} "
                            f"<span style='color:{MUTED}; font-size:0.72rem;'>"
                            f"({obs['source']})</span></div>"
                            f"<div class='obs-assign'>{assign_str}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with btn_col:
                        st.markdown("<div class='obs-remove'>", unsafe_allow_html=True)
                        if st.button(
                            "✕",
                            key=f"rm_{obs['id']}",
                            help="Remove this observation",
                        ):
                            st.session_state.observations = [
                                o for o in st.session_state.observations
                                if o["id"] != obs["id"]
                            ]
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# TAB 4 — Audit trail (full width, grouped tables)
# ---------------------------------------------------------------------------

if active_view == _VIEW_AUDIT:
    st.markdown("<div class='card-title'>Updates by day (injected evidence inputs)</div>",
                unsafe_allow_html=True)
    if not st.session_state.observations:
        st.caption("No observations yet.")
    else:
        update_rows = []
        for obs in sorted(st.session_state.observations, key=lambda o: o["day"]):
            for node, state in obs.get("assignments", {}).items():
                reason = obs.get("per_assignment_reasons", {}).get(node, "")
                update_rows.append({
                    "Day": obs["day"],
                    "Node": node.replace("_", " "),
                    "Injected evidence": state,
                    "Headline / note": obs["headline"],
                    "Rationale": reason,
                    "Source": obs["source"],
                })
            for node, dist in obs.get("soft_assignments", {}).items():
                reason = obs.get("per_assignment_reasons", {}).get(node, "")
                top_state = max(dist, key=dist.get)
                update_rows.append({
                    "Day": obs["day"],
                    "Node": node.replace("_", " "),
                    "Injected evidence": f"{top_state} ({dist[top_state]*100:0.1f}%, soft)",
                    "Headline / note": obs["headline"],
                    "Rationale": reason,
                    "Source": obs["source"],
                })
        st.dataframe(
            pd.DataFrame(update_rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Day": st.column_config.NumberColumn("Day", width="small"),
                "Node": st.column_config.TextColumn("Node", width="medium"),
                "Injected evidence": st.column_config.TextColumn(
                    "Injected evidence", width="medium"
                ),
                "Headline / note": st.column_config.TextColumn(
                    "Headline / note", width="large"),
                "Rationale": st.column_config.TextColumn(
                    "Rationale", width="large"),
                "Source": st.column_config.TextColumn("Source", width="small"),
            },
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Intermediate node marginals</div>"
                "<div class='card-sub'>Grouped by state-set shape so each "
                "table is compact and fully readable.</div>",
                unsafe_allow_html=True)

    intermediate_nodes = [
        "Iran_Aligned_Militia_Attacks", "Tanker_Incidents", "US_Military_Response",
        "Strait_Operationally_Closed", "Energy_Infrastructure_Damage",
        "Conflict_Duration", "Diplomatic_Resolution_Path", "Oil_Price_Regime",
    ]

    groups: Dict[Tuple[str, ...], List[str]] = {}
    for node in intermediate_nodes:
        key = tuple(STATES[node])
        groups.setdefault(key, []).append(node)

    for state_set, group_nodes in groups.items():
        rows = []
        for node in group_nodes:
            marginal = engine.get_node_marginal(node)
            observed_label = evidence.get(node, "")
            if not observed_label and node in soft_evidence:
                dist = soft_evidence[node]
                top_state = max(dist, key=dist.get)
                observed_label = f"{top_state} (soft)"
            row = {
                "Node": node.replace("_", " "),
                "Injected evidence": observed_label,
            }
            for state in state_set:
                row[state] = float(marginal.get(state, 0.0)) * 100
            rows.append(row)
        df = pd.DataFrame(rows, columns=["Node", "Injected evidence", *state_set])
        col_cfg = {
            "Node": st.column_config.TextColumn("Node", width="medium"),
            "Injected evidence": st.column_config.TextColumn(
                "Injected evidence", width="medium"
            ),
        }
        for s in state_set:
            col_cfg[s] = st.column_config.ProgressColumn(
                s, format="%.1f%%", min_value=0.0, max_value=100.0,
            )
        st.dataframe(
            df, hide_index=True, width="stretch",
            column_config=col_cfg,
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.caption(
    "Probabilities are illustrative — CPTs are expert-elicited, not "
    "calibrated from historical data. Translator output is an LLM reading "
    "of each headline and should be reviewed before acting on the resulting "
    "probabilities. See README for the BN-vs-HMM rationale."
)
