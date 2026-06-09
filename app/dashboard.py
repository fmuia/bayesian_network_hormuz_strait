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

from components import (  # noqa: E402
    audit_view, edge_rationale, evolution_chart, network_view,
    observation_log, scenario_cards,
    triage_view,
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
    network_view.render(
        st, all_marginals=all_marginals, evidence=evidence,
        soft_evidence=soft_evidence, node_ci_table=node_ci_table,
        observed_day_map=observed_day_map, topology=TOPOLOGY,
    )

    with st.expander("Appendix — math and implementation details", expanded=False):
        _render_model_appendix()


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
