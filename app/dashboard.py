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
from src.sensitivity import scenario_credible_intervals
from src.translator import (
    TranslatorError,
    TranslatorResult,
    available_providers,
    is_available as translator_available,
    translate_headline,
)
from src.viz import build_agraph_payload, render_network_png

# ---------------------------------------------------------------------------
# Page setup & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Strait of Hormuz — Adaptive Scenario Probabilities",
    layout="wide",
    initial_sidebar_state="expanded",
)

TEAL = "#1A7A6D"
NAVY = "#1B2A3D"
PANEL = "#F5F5F5"
RULE = "#E5E7EB"
MUTED = "#6B7280"
GREEN = "#2E8B57"
AMBER = "#D4A017"
RED = "#B22222"

SCENARIO_COLOR = {
    "Stress_Mitigates": GREEN,
    "Prolonged_Conflict": AMBER,
    "Severe_Closure": RED,
}
SCENARIO_LABEL = {
    "Stress_Mitigates": "Stress Mitigates",
    "Prolonged_Conflict": "Prolonged Conflict",
    "Severe_Closure": "Severe Closure",
}
ROOT_DRIVER_STYLE = {
    "US_Iran_Negotiations": ("#DBEAFE", "#1D4ED8"),
    "Iranian_Regime_Stability": ("#FCE7F3", "#BE185D"),
    "Third_Party_Mediation": ("#FEF3C7", "#B45309"),
    "Sanctions_Trajectory": ("#EDE9FE", "#6D28D9"),
}

st.markdown(
    f"""
    <style>
      html, body, [class*="css"] {{
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        color: {NAVY};
      }}
      /* Keep Streamlit header/toolbar visible so sidebar can always be reopened. */
      [data-testid="stHeader"] {{ background: transparent; }}
      [data-testid="stToolbar"] {{
        display: flex;
        visibility: visible;
      }}
      [data-testid="stDecoration"] {{ display: none; }}
      [data-testid="stStatusWidget"] {{ display: none; }}
      [data-testid="stSidebarCollapsedControl"] {{
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: fixed;
        top: 0.6rem;
        left: 0.7rem;
        z-index: 10000;
        background: white;
        border: 1px solid {RULE};
        border-radius: 8px;
      }}

      /* Align sidebar and main content with the top of the page. */
      .block-container {{
        padding-top: 0.4rem; padding-bottom: 3rem; max-width: 1600px;
      }}
      [data-testid="stAppViewContainer"] > .main {{ padding-top: 0; }}
      [data-testid="stSidebar"] {{ background: {PANEL}; }}
      [data-testid="stSidebar"] > div:first-child {{ padding-top: 0; }}
      [data-testid="stSidebarHeader"] {{
        padding: 0.12rem 0.45rem 0.08rem 0.45rem;
        margin: 0;
        min-height: 1.8rem;
        height: auto;
      }}
      [data-testid="stSidebarCollapseButton"] {{
        display: flex !important;
        visibility: visible !important;
      }}
      section[data-testid="stSidebar"] .block-container {{
        padding-top: 0.22rem;
      }}
      [data-testid="stSidebarUserContent"] {{
        padding-top: 0.22rem;
      }}

      h1, h2, h3, h4 {{ color: {NAVY}; font-weight: 600; }}

      /* Header */
      .demo-title {{
        font-size: 1.45rem; font-weight: 700; color: {NAVY};
        margin: 0 0 0.1rem 0;
      }}
      .demo-subtitle {{
        font-size: 0.88rem; color: {MUTED}; margin-bottom: 0.9rem;
      }}

      /* Reusable "card" container for each main-area object. */
      .card {{
        background: white; border: 1px solid {RULE};
        border-radius: 8px; padding: 1rem 1.2rem;
        box-shadow: 0 1px 2px rgba(27,42,61,0.04);
        margin-bottom: 1rem;
      }}
      .card-title {{
        font-size: 0.72rem; font-weight: 700; color: {TEAL};
        text-transform: uppercase; letter-spacing: 0.08em;
        margin: 0 0 0.65rem 0;
      }}
      .card-sub {{ font-size: 0.82rem; color: {MUTED}; margin-bottom: 0.7rem; }}

      /* Scenario cards — CSS-grid row, no Streamlit columns needed */
      .scenario-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
      }}
      .scenario-card {{
        background: white; border: 1px solid {RULE};
        border-left: 5px solid {NAVY};
        padding: 1rem 1.1rem; border-radius: 6px;
      }}
      .scenario-name {{
        font-size: 0.74rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.06em;
        color: {MUTED}; margin-bottom: 0.2rem;
      }}
      .scenario-prob {{
        font-size: 2.3rem; font-weight: 700; line-height: 1.0;
        margin-bottom: 0.15rem;
      }}
      .scenario-ci {{ font-size: 0.76rem; color: {MUTED}; margin-bottom: 0.55rem; }}
      .scenario-narrative {{ font-size: 0.82rem; color: {NAVY}; line-height: 1.4; }}

      /* Sidebar */
      .sb-provider {{
        display: inline-block; padding: 0.25rem 0.65rem;
        border-radius: 999px; font-size: 0.78rem; font-weight: 600;
        background: #E7F4EF; color: {GREEN};
        margin-bottom: 0.5rem;
      }}
      .sb-header {{
        margin: 0.3rem 0 0.62rem 0;
      }}
      .sb-header-title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: {NAVY};
        line-height: 1.15;
        white-space: nowrap;
      }}
      .sb-header-sub {{
        font-size: 0.74rem;
        color: {MUTED};
        margin-top: 0.1rem;
      }}
      .sb-provider.warn {{ background: #FEF3C7; color: #92400E; }}
      .sb-title {{
        font-size: 0.68rem; font-weight: 700; color: {TEAL};
        text-transform: uppercase; letter-spacing: 0.08em;
        margin: 0.2rem 0 0.35rem 0;
      }}
      .day-pill {{
        display: inline-block; background: {NAVY}; color: white;
        padding: 0.25rem 0.65rem; border-radius: 14px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em;
      }}
      .sb-hint {{ font-size: 0.76rem; color: {MUTED}; margin: 0.25rem 0 0.4rem 0; }}
      .stream-line {{
        font-family: 'JetBrains Mono', Menlo, monospace;
        font-size: 0.78rem; color: {NAVY};
        background: white; border: 1px solid {RULE}; border-radius: 6px;
        padding: 0.45rem 0.6rem;
        margin-top: 0.3rem; white-space: pre-wrap; word-break: break-word;
      }}
      .stream-done {{ border-color: {GREEN}; background: #F0FAF5; color: {GREEN}; }}
      .stream-error {{ border-color: {RED}; background: #FEF2F2; color: {RED}; }}

      /* Translator output panel */
      .translator-headline {{
        font-size: 0.95rem; font-weight: 600; color: {NAVY};
        margin-bottom: 0.3rem;
      }}
      .translator-rationale {{
        font-size: 0.83rem; color: #4B5563; margin: 0.25rem 0 0.65rem 0;
        font-style: italic; line-height: 1.4;
      }}
      .assign-chip {{
        display: inline-block; padding: 0.22rem 0.6rem;
        border-radius: 12px; background: #EAF4F2; color: {TEAL};
        font-size: 0.78rem; font-weight: 600; margin: 0.15rem 0.3rem 0.15rem 0;
      }}
      .root-chip {{
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 600;
        margin: 0.15rem 0.35rem 0.2rem 0;
      }}
      .meta {{
        font-size: 0.7rem; color: #9CA3AF; margin-top: 0.55rem;
        text-align: right;
      }}

      /* Day-grouped log */
      .log-scroll {{ max-height: 540px; overflow-y: auto; padding-right: 0.3rem; }}
      .day-block {{
        background: white; border: 1px solid {RULE}; border-radius: 6px;
        padding: 0.7rem 0.9rem; margin-bottom: 0.55rem;
      }}
      .day-block-header {{
        font-size: 0.73rem; font-weight: 700; color: {TEAL};
        text-transform: uppercase; letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
      }}
      .obs-row {{
        font-size: 0.82rem; color: {NAVY};
        padding: 0.35rem 0; border-top: 1px solid {RULE};
      }}
      .obs-row:first-of-type {{ border-top: none; padding-top: 0.1rem; }}
      .obs-headline {{ font-weight: 500; }}
      .obs-assign {{ color: {MUTED}; font-size: 0.78rem; margin-top: 0.1rem; }}

      /* Tabs a little tighter & more legible */
      div[data-baseweb="tab-list"] button {{ font-weight: 600; }}

      /* Model explanation */
      .explain h4 {{ margin: 0.6rem 0 0.25rem 0; font-size: 0.95rem; }}
      .explain p  {{ font-size: 0.87rem; color: {NAVY}; line-height: 1.5;
                     margin: 0 0 0.55rem 0; }}
      .explain ul {{ margin: 0 0 0.6rem 1rem; padding: 0; }}
      .explain li {{ font-size: 0.86rem; color: {NAVY}; line-height: 1.45;
                     margin-bottom: 0.15rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource
def get_engine() -> BNInferenceEngine:
    return BNInferenceEngine(build_network())


@st.cache_data(show_spinner=False)
def cached_credible_intervals(
    evidence_items: Tuple[Tuple[str, str], ...],
) -> Dict[str, Tuple[float, float, float]]:
    return scenario_credible_intervals(dict(evidence_items), m=200, concentration=20.0)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_SS_DEFAULTS = {
    "observations": [],
    "current_day": 1,
    "last_translation": None,
    "translator_error": None,
    "translator_raw": "",
    "pending_headline": None,
    "selected_node": None,
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_SESSION_STORE = ROOT / "data" / "dashboard_saved_sessions.json"


def _load_session_store() -> Dict[str, Dict]:
    if not _SESSION_STORE.exists():
        return {}
    try:
        return json.loads(_SESSION_STORE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_session_store(store: Dict[str, Dict]) -> None:
    _SESSION_STORE.parent.mkdir(parents=True, exist_ok=True)
    _SESSION_STORE.write_text(
        json.dumps(store, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _snapshot_session_state() -> Dict:
    return {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "current_day": st.session_state.current_day,
        "observations": st.session_state.observations,
        "last_translation": st.session_state.last_translation,
        "translator_error": st.session_state.translator_error,
        "translator_raw": st.session_state.translator_raw,
        "selected_node": st.session_state.selected_node,
    }


def _save_named_session(name: str) -> None:
    store = _load_session_store()
    store[name] = _snapshot_session_state()
    _write_session_store(store)


def _restore_named_session(name: str) -> bool:
    store = _load_session_store()
    payload = store.get(name)
    if payload is None:
        return False
    for key in _SS_DEFAULTS:
        st.session_state[key] = payload.get(key, _SS_DEFAULTS[key])
    return True


def _delete_named_session(name: str) -> bool:
    store = _load_session_store()
    if name not in store:
        return False
    del store[name]
    _write_session_store(store)
    return True


def _append_observation(
    headline: str,
    assignments: Dict[str, str],
    soft_assignments: Optional[Dict[str, Dict[str, float]]] = None,
    rationale: str = "",
    per_assignment_reasons: Optional[Dict[str, str]] = None,
    source: str = "translator",
) -> None:
    obs = Observation(
        day=st.session_state.current_day,
        headline=headline,
        assignments=dict(assignments),
        soft_assignments=dict(soft_assignments or {}),
        rationale=rationale,
        per_assignment_reasons=per_assignment_reasons or {},
        source=source,
    )
    st.session_state.observations.append({"id": uuid.uuid4().hex, **asdict(obs)})


def _merged_evidence() -> Tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
    """Latest observation wins on conflict, in insertion order."""
    hard_merged: Dict[str, str] = {}
    soft_merged: Dict[str, Dict[str, float]] = {}
    for obs in st.session_state.observations:
        for node, state in obs.get("assignments", {}).items():
            hard_merged[node] = state
            soft_merged.pop(node, None)
        for node, dist in obs.get("soft_assignments", {}).items():
            soft_merged[node] = {k: float(v) for k, v in dist.items()}
            hard_merged.pop(node, None)
    return hard_merged, soft_merged


def _render_model_overview() -> None:
    st.markdown(
        "<div class='explain'>"
        "<p>The Bayesian network encodes qualitative causal structure "
        "between four <b>root drivers</b> (negotiations, regime "
        "stability, third-party mediation, sanctions trajectory), "
        "<b>eight intermediate nodes</b> (proxy activity, tanker "
        "incidents, US military response, strait closure, energy "
        "infrastructure damage, conflict duration, diplomatic path, "
        "oil price regime), and a terminal <b>Scenario</b> node "
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

        Credible intervals are estimated by resampling CPT columns from Dirichlet distributions and rerunning inference:

        $$
        \theta_{j,\cdot}^{(m)} \sim \text{Dirichlet}(\alpha_{j,\cdot})
        $$

        with concentration set to 20 and $m=200$ draws in the dashboard.
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


def _run_translator(headline: str, stream_slot) -> None:
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

    try:
        result: TranslatorResult = translate_headline(headline, on_step=on_step)
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
    }
    if result.assignments:
        soft_assignments = {
            a.node: dict(a.state_probs) for a in result.assignments
        }
        _append_observation(
            headline=result.headline,
            assignments={},
            soft_assignments=soft_assignments,
            rationale=result.rationale,
            per_assignment_reasons={a.node: a.reason for a in result.assignments},
            source="translator",
        )
        _write(
            "done", "validated",
            f"{len(result.assignments)} assignment(s) · model {result.model}",
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

translator_on = translator_available()
providers = available_providers()
provider_labels = {"claude-code": "Claude Code", "openai": "OpenAI API"}

with st.sidebar:
    st.markdown(
        "<div class='sb-header'>"
        "<div class='sb-header-title'>Scenario Session Controls</div>"
        "<div class='sb-header-sub'>Translator, observations, and state</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # -- Provider chip (one line) ------------------------------------------
    if translator_on:
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

    st.markdown("<div class='sb-title'>Translate a headline</div>",
                unsafe_allow_html=True)

    with st.form("headline_form", clear_on_submit=True):
        headline_input = st.text_area(
            "News headline",
            placeholder="e.g. 'Iran suspends Hormuz traffic inspections'",
            height=72,
            disabled=not translator_on,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(
            "Translate & observe", type="primary",
            disabled=not translator_on, width="stretch",
        )
        if submitted and headline_input.strip():
            st.session_state.pending_headline = headline_input.strip()

    # Stream slot lives just below the form — compact, single-line.
    stream_slot = st.empty()

    # Run translator *after* the slot is in the sidebar, so updates appear here.
    if st.session_state.pending_headline is not None:
        headline = st.session_state.pending_headline
        st.session_state.pending_headline = None
        _run_translator(headline, stream_slot)

    with st.expander("Examples", expanded=False):
        for idx, ex in enumerate(EXAMPLE_HEADLINES):
            if st.button(
                ex.text, key=f"ex_{idx}",
                width="stretch", disabled=not translator_on,
            ):
                st.session_state.pending_headline = ex.text
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

engine = get_engine()
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
    ci_table = cached_credible_intervals(tuple(sorted(ci_evidence.items())))

all_marginals = {n: engine.get_node_marginal(n) for n in STATES}

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

st.markdown(
    "<div class='demo-title'>Adaptive Scenario Probability Framework — Strait of Hormuz</div>",
    unsafe_allow_html=True,
)
with st.expander("How this model works", expanded=False):
    _render_model_overview()


# ===========================================================================
# PINNED TOP BAND — scenario cards + probability evolution
# ===========================================================================

with st.container(border=True):
    st.markdown("<div class='card-title'>Scenario outlook</div>",
                unsafe_allow_html=True)

    cards_html = "<div class='scenario-grid'>"
    for scenario in ["Stress_Mitigates", "Prolonged_Conflict", "Severe_Closure"]:
        mean, lo, hi = ci_table[scenario]
        color = SCENARIO_COLOR[scenario]
        label = SCENARIO_LABEL[scenario]
        narrative = SCENARIO_NARRATIVES[scenario]
        cards_html += (
            f"<div class='scenario-card' style='border-left-color:{color};'>"
            f"  <div class='scenario-name' style='color:{color};'>{label}</div>"
            f"  <div class='scenario-prob' style='color:{color};'>{mean*100:0.1f}%</div>"
            f"  <div class='scenario-ci'>80% CI: {lo*100:0.1f}% – {hi*100:0.1f}%</div>"
            f"  <div class='scenario-narrative'>{narrative}</div>"
            f"</div>"
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    with st.expander("Uncertainty detail — 80% credible intervals", expanded=False):
        ci_df = pd.DataFrame([
            {
                "Scenario": SCENARIO_LABEL[s],
                "Mean": ci_table[s][0],
                "Lo": ci_table[s][1],
                "Hi": ci_table[s][2],
            }
            for s in ["Stress_Mitigates", "Prolonged_Conflict", "Severe_Closure"]
        ])
        ci_scale = alt.Scale(
            domain=[SCENARIO_LABEL[s] for s in
                    ["Stress_Mitigates", "Prolonged_Conflict", "Severe_Closure"]],
            range=[GREEN, AMBER, RED],
        )
        err_rule = alt.Chart(ci_df).mark_rule(strokeWidth=4).encode(
            y=alt.Y("Scenario:N", sort=None, title=None,
                    axis=alt.Axis(labelColor=NAVY)),
            x=alt.X("Lo:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%", labelColor=NAVY,
                                  titleColor=NAVY),
                    title="Probability"),
            x2="Hi:Q",
            color=alt.Color("Scenario:N", scale=ci_scale, legend=None),
        )
        err_caps_lo = alt.Chart(ci_df).mark_tick(
            thickness=3, size=18
        ).encode(
            y=alt.Y("Scenario:N", sort=None),
            x="Lo:Q",
            color=alt.Color("Scenario:N", scale=ci_scale, legend=None),
        )
        err_caps_hi = alt.Chart(ci_df).mark_tick(
            thickness=3, size=18
        ).encode(
            y=alt.Y("Scenario:N", sort=None),
            x="Hi:Q",
            color=alt.Color("Scenario:N", scale=ci_scale, legend=None),
        )
        err_pts = alt.Chart(ci_df).mark_circle(size=180, opacity=1).encode(
            y=alt.Y("Scenario:N", sort=None),
            x="Mean:Q",
            color=alt.Color("Scenario:N", scale=ci_scale, legend=None),
            tooltip=[
                alt.Tooltip("Scenario:N"),
                alt.Tooltip("Mean:Q", format=".1%", title="Mean"),
                alt.Tooltip("Lo:Q", format=".1%", title="Lo (10%)"),
                alt.Tooltip("Hi:Q", format=".1%", title="Hi (90%)"),
            ],
        )
        err_chart = (
            err_rule + err_caps_lo + err_caps_hi + err_pts
        ).properties(height=170).configure_view(stroke=None)
        st.altair_chart(err_chart, use_container_width=True)
        st.caption(
            "Intervals come from resampling CPT parameters (Dirichlet, "
            "concentration = 20, m = 200) and re-running inference."
        )


# ---- Probability evolution (Altair, interactive) -------------------------

with st.container(border=True):
    st.markdown("<div class='card-title'>Probability evolution by day</div>",
                unsafe_allow_html=True)

    if not st.session_state.observations:
        st.markdown(
            f"<div class='card-sub' style='color:{MUTED};'>"
            "No observations yet — the timeline fills as you translate or "
            "override observations.</div>",
            unsafe_allow_html=True,
        )
    else:
        history_rows: List[Dict] = []
        engine_h = get_engine()
        engine_h.clear_evidence()
        priors = engine_h.get_prior_probabilities()
        history_rows.append({"Day": 0, "HeadlinesOnDay": "(prior)",
                             "n_obs": 0, **priors})

        grouped: Dict[int, List[Dict]] = {}
        for obs in st.session_state.observations:
            grouped.setdefault(obs["day"], []).append(obs)

        cum_hard: Dict[str, str] = {}
        cum_soft: Dict[str, Dict[str, float]] = {}
        for day in sorted(grouped):
            day_obs = grouped[day]
            for obs in day_obs:
                for node, state in obs.get("assignments", {}).items():
                    cum_hard[node] = state
                    cum_soft.pop(node, None)
                for node, dist in obs.get("soft_assignments", {}).items():
                    cum_soft[node] = {k: float(v) for k, v in dist.items()}
                    cum_hard.pop(node, None)
            engine_h.clear_evidence()
            if cum_hard:
                engine_h.update_evidence(cum_hard)
            if cum_soft:
                engine_h.update_soft_evidence(cum_soft)
            headlines = " · ".join(o["headline"] for o in day_obs)
            if len(headlines) > 180:
                headlines = headlines[:177] + "…"
            history_rows.append({
                "Day": day,
                "HeadlinesOnDay": headlines,
                "n_obs": len(day_obs),
                **engine_h.get_scenario_probabilities(),
            })

        wide = pd.DataFrame(history_rows)
        long_rows = []
        for _, r in wide.iterrows():
            for sc in ["Stress_Mitigates", "Prolonged_Conflict", "Severe_Closure"]:
                long_rows.append({
                    "Day": int(r["Day"]),
                    "Scenario": SCENARIO_LABEL[sc],
                    "ScenarioKey": sc,
                    "Probability": float(r[sc]),
                    "HeadlinesOnDay": r["HeadlinesOnDay"],
                    "n_obs": int(r["n_obs"]),
                })
        long_df = pd.DataFrame(long_rows)

        color_scale = alt.Scale(
            domain=[SCENARIO_LABEL[s] for s in
                    ["Stress_Mitigates", "Prolonged_Conflict", "Severe_Closure"]],
            range=[GREEN, AMBER, RED],
        )

        base = alt.Chart(long_df).encode(
            x=alt.X("Day:O", title="Day",
                    axis=alt.Axis(
                        labelColor=NAVY,
                        titleColor=NAVY,
                        labelAngle=0,
                    )),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                    title="Probability",
                    axis=alt.Axis(format="%", labelColor=NAVY, titleColor=NAVY)),
            color=alt.Color("Scenario:N", scale=color_scale,
                            legend=alt.Legend(title=None, orient="top")),
        )
        lines = base.mark_line(strokeWidth=2.6, point=alt.OverlayMarkDef(size=70))
        hover = alt.selection_point(
            fields=["Day"], nearest=True, on="mouseover", empty=False,
        )
        tooltip = base.mark_circle(size=120, opacity=0).encode(
            tooltip=[
                alt.Tooltip("Day:O"),
                alt.Tooltip("Scenario:N"),
                alt.Tooltip("Probability:Q", format=".1%"),
                alt.Tooltip("n_obs:Q", title="# obs added"),
                alt.Tooltip("HeadlinesOnDay:N", title="Headlines"),
            ],
        ).add_params(hover)
        chart = (lines + tooltip).properties(height=260).configure_view(
            stroke=None,
        )
        st.altair_chart(chart, use_container_width=True)

        last_day = max(grouped)
        last_headlines = " · ".join(o["headline"] for o in grouped[last_day])
        st.caption(
            f"Most recent update — Day {last_day}: {last_headlines}"
        )


# ===========================================================================
# TABS — Network & model / Observations / Audit trail
# ===========================================================================

tab_net, tab_obs, tab_audit = st.tabs(
    ["🕸️  Network & model", "📝  Observations", "🔎  Audit trail"]
)


# ---------------------------------------------------------------------------
# TAB 1 — Network & model (interactive graph + click-to-override / explain)
# ---------------------------------------------------------------------------

with tab_net:
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
            nodes, edges, config = build_agraph_payload(
                all_marginals,
                observed=evidence,
                observed_day=observed_day_map,
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
                st.markdown(
                    f"<div class='card-sub'><b>{sel.replace('_',' ')}</b></div>",
                    unsafe_allow_html=True,
                )
                marginal = all_marginals[sel]
                if sel in evidence:
                    st.markdown(
                        f"<div class='card-sub'>Currently observed: "
                        f"<b>{evidence[sel]}</b> (day {observed_day_map.get(sel, '?')})"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                elif sel in soft_evidence:
                    dist = soft_evidence[sel]
                    top_state = max(dist, key=dist.get)
                    st.markdown(
                        f"<div class='card-sub'>Soft evidence from headlines: "
                        f"<b>{top_state}</b> ({dist[top_state]*100:0.1f}%, "
                        f"day {observed_day_map.get(sel, '?')})</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='card-sub'>Posterior marginal after propagating "
                        "all injected evidence:</div>",
                        unsafe_allow_html=True,
                    )
                for state, prob in marginal.items():
                    st.markdown(
                        f"<div style='font-size:0.8rem; color:{NAVY}; "
                        f"margin-bottom:0.15rem;'>{state} — "
                        f"<b>{prob*100:0.1f}%</b></div>",
                        unsafe_allow_html=True,
                    )
                    st.progress(min(max(prob, 0.0), 1.0))
            else:
                st.markdown(
                    "<div class='card-sub'>Click a node in the graph to inspect "
                    "its posterior distribution.</div>",
                    unsafe_allow_html=True,
                )

        with st.container(border=True):
            st.markdown(
                "<div class='card-title'>Override</div>"
                "<div class='card-sub'>Set this node directly to create a "
                "manual observation on the current day.</div>",
                unsafe_allow_html=True,
            )
            if sel and sel in STATES:
                with st.form(f"override_{sel}", clear_on_submit=False):
                    chosen = st.radio(
                        "State",
                        STATES[sel],
                        horizontal=True,
                        key=f"radio_{sel}",
                    )
                    note = st.text_input(
                        "Note (optional)", key=f"note_{sel}",
                        placeholder="What drove this override?",
                    )
                    ok = st.form_submit_button("Set observation", type="primary")
                    if ok:
                        _append_observation(
                            headline=note.strip() or f"Manual: {sel} = {chosen}",
                            assignments={sel: chosen},
                            rationale="Set directly by the analyst via the network.",
                            per_assignment_reasons={sel: "Manual override."},
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
# TAB 2 — Observations (latest translation + day-grouped log)
# ---------------------------------------------------------------------------

with tab_obs:
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
            st.markdown(
                f"""
                <div class='translator-headline'>“{t['headline']}”</div>
                <div class='translator-rationale'>{t['rationale']}</div>
                <div>{chips_html}</div>
                <div class='meta'>provider: {t.get('provider','?')} ·
                model: {t['model']}</div>
                """,
                unsafe_allow_html=True,
            )
            if t["assignments"]:
                with st.expander("Per-assignment evidence input (translator soft evidence)"):
                    for a in t["assignments"]:
                        probs = a.get("state_probs", {})
                        probs_text = " · ".join(
                            f"{k.replace('_',' ')}: {float(v)*100:0.1f}%"
                            for k, v in probs.items()
                        )
                        probs_suffix = f"  \n  {probs_text}" if probs_text else ""
                        st.markdown(
                            f"- **{a['node'].replace('_',' ')} = "
                            f"`{a['state']}`** — {a['reason']}"
                            f"{probs_suffix}"
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
                rows_html = ""
                for obs in day_obs:
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
                    rows_html += (
                        f"<div class='obs-row'>"
                        f"<div class='obs-headline'>{obs['headline']} "
                        f"<span style='color:{MUTED}; font-size:0.72rem;'>"
                        f"({obs['source']})</span></div>"
                        f"<div class='obs-assign'>{assign_str}</div>"
                        f"</div>"
                    )
                st.markdown(
                    f"<div class='day-block'>"
                    f"<div class='day-block-header'>Day {day} · "
                    f"{len(day_obs)} observation(s)</div>"
                    f"{rows_html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                btn_cols = st.columns(min(len(day_obs), 4))
                for i, obs in enumerate(day_obs):
                    with btn_cols[i % len(btn_cols)]:
                        if st.button(
                            f"× remove #{i+1}",
                            key=f"rm_{obs['id']}",
                            width="stretch",
                        ):
                            st.session_state.observations = [
                                o for o in st.session_state.observations
                                if o["id"] != obs["id"]
                            ]
                            st.rerun()


# ---------------------------------------------------------------------------
# TAB 3 — Audit trail (full width, grouped tables)
# ---------------------------------------------------------------------------

with tab_audit:
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
            use_container_width=True,
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
        "Iranian_Proxy_Activity", "Tanker_Incidents", "US_Military_Response",
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
                observed_label = f"{top_state} ({dist[top_state]*100:0.0f}%, soft)"
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
            df, hide_index=True, use_container_width=True,
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
