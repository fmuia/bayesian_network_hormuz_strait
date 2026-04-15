"""Streamlit dashboard for the Strait of Hormuz BN demo.

Run with: ``streamlit run app/dashboard.py`` (or ``pixi run app``).

The workflow is intentionally two-layer:

    Free-text headline
        -> Translation layer (OpenAI)
        -> {node: state, ...} observation
        -> Bayesian network inference
        -> Scenario probabilities.

If ``OPENAI_API_KEY`` is missing, the translator is disabled and a
manual node/state picker stands in for it so the demo still runs.
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make the project root importable when launched via `streamlit run`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

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
from src.viz import render_network

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

st.markdown(
    f"""
    <style>
      html, body, [class*="css"] {{
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        color: {NAVY};
      }}
      .block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px; }}
      h1, h2, h3, h4 {{ color: {NAVY}; font-weight: 600; }}
      .demo-title {{
        font-size: 1.75rem; font-weight: 700; color: {NAVY};
        margin-bottom: 0.1rem;
      }}
      .demo-subtitle {{
        font-size: 0.95rem; color: #4B5563; margin-bottom: 0.9rem;
      }}
      .arch-banner {{
        background: #EAF4F2; border-left: 3px solid {TEAL};
        padding: 0.65rem 0.95rem; border-radius: 4px;
        font-size: 0.87rem; color: {NAVY}; margin-bottom: 1.1rem;
      }}
      .arch-banner b {{ color: {TEAL}; }}
      .scenario-card {{
        background: white; border: 1px solid #E5E7EB;
        border-left: 5px solid {NAVY};
        padding: 1rem 1.1rem; border-radius: 4px; height: 100%;
        box-shadow: 0 1px 2px rgba(27,42,61,0.04);
      }}
      .scenario-name {{
        font-size: 0.85rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.04em;
        color: #6B7280; margin-bottom: 0.2rem;
      }}
      .scenario-prob {{
        font-size: 2.4rem; font-weight: 700; line-height: 1.0;
        margin-bottom: 0.15rem;
      }}
      .scenario-ci {{ font-size: 0.8rem; color: #6B7280; margin-bottom: 0.55rem; }}
      .scenario-narrative {{ font-size: 0.85rem; color: {NAVY}; line-height: 1.35; }}
      .section-title {{
        font-size: 1.05rem; font-weight: 600; color: {NAVY};
        margin: 1.3rem 0 0.5rem 0;
      }}
      .sidebar-header {{
        font-size: 1.1rem; font-weight: 700; color: {NAVY};
        margin: 0.2rem 0 0.15rem 0;
      }}
      .sidebar-hint {{ font-size: 0.8rem; color: #6B7280; margin-bottom: 0.7rem; }}
      [data-testid="stSidebar"] {{ background: {PANEL}; }}
      .translator-panel {{
        background: white; border: 1px solid #E5E7EB;
        border-radius: 4px; padding: 0.85rem 1rem;
        margin-top: 0.6rem;
      }}
      .translator-headline {{
        font-size: 0.9rem; font-weight: 600; color: {NAVY};
      }}
      .translator-rationale {{
        font-size: 0.82rem; color: #4B5563; margin: 0.3rem 0 0.55rem 0;
        font-style: italic;
      }}
      .assign-chip {{
        display: inline-block; padding: 0.18rem 0.55rem;
        border-radius: 12px; background: #EAF4F2; color: {TEAL};
        font-size: 0.78rem; font-weight: 600; margin: 0.1rem 0.25rem 0.1rem 0;
      }}
      .log-row {{
        background: white; border: 1px solid #E5E7EB; border-radius: 4px;
        padding: 0.55rem 0.75rem; margin-bottom: 0.4rem;
      }}
      .log-day {{
        font-size: 0.75rem; font-weight: 700; color: {TEAL};
        letter-spacing: 0.05em; text-transform: uppercase;
      }}
      .log-headline {{ font-size: 0.87rem; color: {NAVY}; font-weight: 500; }}
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
# Session state init
# ---------------------------------------------------------------------------

if "observations" not in st.session_state:
    st.session_state.observations: List[Dict] = []
if "day_counter" not in st.session_state:
    st.session_state.day_counter = 0
if "last_translation" not in st.session_state:
    st.session_state.last_translation: Optional[Dict] = None
if "translator_error" not in st.session_state:
    st.session_state.translator_error: Optional[str] = None
if "pending_headline" not in st.session_state:
    st.session_state.pending_headline: Optional[str] = None


def _append_observation(
    headline: str,
    assignments: Dict[str, str],
    rationale: str = "",
    per_assignment_reasons: Optional[Dict[str, str]] = None,
    source: str = "translator",
) -> None:
    st.session_state.day_counter += 1
    obs = Observation(
        day=st.session_state.day_counter,
        headline=headline,
        assignments=dict(assignments),
        rationale=rationale,
        per_assignment_reasons=per_assignment_reasons or {},
        source=source,
    )
    st.session_state.observations.append({"id": uuid.uuid4().hex, **asdict(obs)})


def _merged_evidence() -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for obs in st.session_state.observations:
        merged.update(obs["assignments"])
    return merged


def _run_translator(headline: str) -> None:
    try:
        result: TranslatorResult = translate_headline(headline)
    except TranslatorError as exc:
        st.session_state.translator_error = str(exc)
        st.session_state.last_translation = None
        return
    st.session_state.translator_error = None
    st.session_state.last_translation = {
        "headline": result.headline,
        "assignments": [asdict(a) for a in result.assignments],
        "rationale": result.rationale,
        "model": result.model,
        "provider": result.provider,
    }
    if result.assignments:
        _append_observation(
            headline=result.headline,
            assignments=result.as_evidence_dict(),
            rationale=result.rationale,
            per_assignment_reasons={a.node: a.reason for a in result.assignments},
            source="translator",
        )
    else:
        st.session_state.translator_error = (
            "Translator returned no assignments for this headline. Try a "
            "more specific headline or use the manual picker."
        )


# Process any headline queued from a previous rerun (example button click
# or form submission). This must run before we draw the sidebar widgets
# so the log and translator panel reflect the latest state on this pass.
if st.session_state.pending_headline is not None:
    _run_translator(st.session_state.pending_headline)
    st.session_state.pending_headline = None


# ---------------------------------------------------------------------------
# Sidebar — headline input + manual fallback + observation log
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    "<div class='sidebar-header'>Simulate incoming news</div>"
    "<div class='sidebar-hint'>Type a headline. The translator layer "
    "extracts which nodes it constrains and feeds them to the Bayesian "
    "network as observed evidence.</div>",
    unsafe_allow_html=True,
)

translator_on = translator_available()
providers = available_providers()
provider_labels = {"claude-code": "Claude Code (subscription)", "openai": "OpenAI API"}
if not translator_on:
    st.sidebar.warning(
        "No translator backend available. Either sign in to Claude Code "
        "on this machine **or** export `OPENAI_API_KEY`, then restart. "
        "Use the **Manual observation** picker below to continue offline.",
        icon="⚠️",
    )
else:
    active_label = provider_labels[providers[0]]
    extra = ""
    if len(providers) > 1:
        extra = f" · fallback: {provider_labels[providers[1]]}"
    st.sidebar.success(
        f"Translator: **{active_label}**{extra}",
        icon="✅",
    )

with st.sidebar.form("headline_form", clear_on_submit=True):
    headline_input = st.text_area(
        "News headline",
        placeholder="e.g. 'Iran announces suspension of Hormuz traffic inspections'",
        height=80,
        disabled=not translator_on,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button(
        "Translate & observe",
        type="primary",
        disabled=not translator_on,
        width="stretch",
    )
    if submitted and headline_input.strip():
        st.session_state.pending_headline = headline_input.strip()
        st.rerun()

# Example headlines — still flow through the translator.
st.sidebar.markdown(
    "<div class='sidebar-hint' style='margin-top:0.5rem;'>Or inject an example:</div>",
    unsafe_allow_html=True,
)
for idx, ex in enumerate(EXAMPLE_HEADLINES):
    if st.sidebar.button(
        ex.text,
        key=f"ex_{idx}",
        width="stretch",
        disabled=not translator_on,
    ):
        st.session_state.pending_headline = ex.text
        st.rerun()

# Manual observation fallback (always available).
with st.sidebar.expander("Manual observation (bypass translator)", expanded=not translator_on):
    observable_nodes = [n for n in STATES.keys() if n != "Scenario"]
    m_node = st.selectbox("Node", observable_nodes, key="m_node")
    m_state = st.selectbox("State", STATES[m_node], key="m_state")
    m_note = st.text_input(
        "Note / headline (optional)",
        key="m_note",
        placeholder="What drove this observation?",
    )
    if st.button("Add observation", key="m_add", width="stretch"):
        _append_observation(
            headline=m_note.strip() or f"Manual: {m_node} = {m_state}",
            assignments={m_node: m_state},
            rationale="Set directly by the analyst (no translator).",
            per_assignment_reasons={m_node: "Manual override."},
            source="manual",
        )
        st.rerun()

# Session controls.
st.sidebar.markdown("---")
if st.sidebar.button("Reset session", width="stretch"):
    st.session_state.observations = []
    st.session_state.day_counter = 0
    st.session_state.last_translation = None
    st.session_state.translator_error = None
    st.rerun()


# ---------------------------------------------------------------------------
# Main compute
# ---------------------------------------------------------------------------

engine = get_engine()
engine.clear_evidence()
evidence = _merged_evidence()
if evidence:
    engine.update_evidence(evidence)

scenario_probs = engine.get_scenario_probabilities()
with st.spinner("Quantifying parameter uncertainty (200 Monte-Carlo runs)…"):
    ci_table = cached_credible_intervals(tuple(sorted(evidence.items())))


# ---------------------------------------------------------------------------
# Header + architecture banner
# ---------------------------------------------------------------------------

st.markdown(
    "<div class='demo-title'>Adaptive Scenario Probability Framework "
    "&mdash; Strait of Hormuz Demo</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='demo-subtitle'>Scenario probabilities update as news "
    "headlines arrive. Bands are 80% credible intervals from second-order "
    "parameter uncertainty.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='arch-banner'>"
    "<b>Two-layer architecture.</b> "
    "<b>Layer 1 — Translator:</b> an LLM reads the headline and proposes "
    "which nodes it constrains. "
    "<b>Layer 2 — Bayesian network:</b> runs inference on those "
    "observations to produce scenario probabilities. "
    "Analyst oversight: every translation is shown with its rationale "
    "and can be removed from the log on the left."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Scenario cards
# ---------------------------------------------------------------------------

card_cols = st.columns(3)
for col, scenario in zip(card_cols, ["Stress_Mitigates", "Prolonged_Conflict", "Severe_Closure"]):
    mean, lo, hi = ci_table[scenario]
    color = SCENARIO_COLOR[scenario]
    label = SCENARIO_LABEL[scenario]
    narrative = SCENARIO_NARRATIVES[scenario]
    with col:
        st.markdown(
            f"""
            <div class='scenario-card' style='border-left-color:{color};'>
              <div class='scenario-name' style='color:{color};'>{label}</div>
              <div class='scenario-prob' style='color:{color};'>{mean*100:0.1f}%</div>
              <div class='scenario-ci'>80% credible interval: {lo*100:0.1f}% – {hi*100:0.1f}%</div>
              <div class='scenario-narrative'>{narrative}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Two-column row: translator output + observation log
# ---------------------------------------------------------------------------

st.markdown("<div class='section-title'>Most recent translation &amp; observation log</div>",
            unsafe_allow_html=True)

trans_col, log_col = st.columns([1.0, 1.1])

with trans_col:
    st.markdown("**Translator output**")
    if st.session_state.translator_error:
        st.error(st.session_state.translator_error)
    elif st.session_state.last_translation is None:
        st.caption("No headline translated yet this session.")
    else:
        t = st.session_state.last_translation
        chips_html = "".join(
            f"<span class='assign-chip'>{a['node'].replace('_',' ')} = {a['state']}</span>"
            for a in t["assignments"]
        ) or "<span style='color:#9CA3AF;'>No assignments</span>"
        st.markdown(
            f"""
            <div class='translator-panel'>
              <div class='translator-headline'>“{t['headline']}”</div>
              <div class='translator-rationale'>{t['rationale']}</div>
              <div>{chips_html}</div>
              <div style='font-size:0.7rem; color:#9CA3AF; margin-top:0.45rem;'>
                provider: {t.get('provider','?')} · model: {t['model']}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if t["assignments"]:
            with st.expander("Per-assignment reasoning"):
                for a in t["assignments"]:
                    st.markdown(
                        f"- **{a['node'].replace('_',' ')} = `{a['state']}`** — {a['reason']}"
                    )

with log_col:
    st.markdown(f"**Observation log** ({len(st.session_state.observations)} active)")
    if not st.session_state.observations:
        st.caption("Translate a headline (or add a manual observation) to begin.")
    else:
        for obs in list(st.session_state.observations):
            assign_str = " · ".join(
                f"{n.replace('_',' ')} = {s}" for n, s in obs["assignments"].items()
            )
            row_l, row_r = st.columns([6, 1])
            with row_l:
                st.markdown(
                    f"""
                    <div class='log-row'>
                      <div class='log-day'>Day {obs['day']} · {obs['source']}</div>
                      <div class='log-headline'>{obs['headline']}</div>
                      <div style='font-size:0.78rem; color:#4B5563; margin-top:0.2rem;'>{assign_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with row_r:
                if st.button("Remove", key=f"rm_{obs['id']}", width="stretch"):
                    st.session_state.observations = [
                        o for o in st.session_state.observations if o["id"] != obs["id"]
                    ]
                    st.rerun()


# ---------------------------------------------------------------------------
# Probability evolution chart
# ---------------------------------------------------------------------------

st.markdown("<div class='section-title'>Probability evolution across the session</div>",
            unsafe_allow_html=True)

if not st.session_state.observations:
    st.info("No observations yet — the chart will populate as you translate headlines.")
else:
    history_rows: List[Dict] = []
    running: Dict[str, str] = {}
    engine_h = get_engine()

    # Prior row (day 0).
    engine_h.clear_evidence()
    priors = engine_h.get_prior_probabilities()
    history_rows.append({"Day": 0, **priors})
    for obs in st.session_state.observations:
        running.update(obs["assignments"])
        engine_h.clear_evidence()
        engine_h.update_evidence(running)
        probs = engine_h.get_scenario_probabilities()
        history_rows.append({"Day": obs["day"], **probs})

    df = pd.DataFrame(history_rows)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for scenario in ["Stress_Mitigates", "Prolonged_Conflict", "Severe_Closure"]:
        ax.plot(
            df["Day"], df[scenario],
            marker="o", linewidth=2.4, markersize=6,
            color=SCENARIO_COLOR[scenario], label=SCENARIO_LABEL[scenario],
        )
    ax.set_xlabel("Day", fontsize=10, color=NAVY)
    ax.set_ylabel("Probability", fontsize=10, color=NAVY)
    ax.set_ylim(0, 1)
    ax.set_xticks(df["Day"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=NAVY, labelsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=3)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


# ---------------------------------------------------------------------------
# Intermediate marginals
# ---------------------------------------------------------------------------

st.markdown("<div class='section-title'>Intermediate-node marginals given current evidence</div>",
            unsafe_allow_html=True)

intermediate_nodes = [
    "Iranian_Proxy_Activity", "Tanker_Incidents", "US_Military_Response",
    "Strait_Operationally_Closed", "Energy_Infrastructure_Damage",
    "Conflict_Duration", "Diplomatic_Resolution_Path", "Oil_Price_Regime",
]

rows = []
for node in intermediate_nodes:
    marginal = engine.get_node_marginal(node)
    row = {"Node": node.replace("_", " "), "Observed": "✓" if node in evidence else ""}
    for state in STATES[node]:
        row[state] = f"{marginal[state]*100:0.1f}%"
    rows.append(row)
all_states = sorted({k for r in rows for k in r if k not in ("Node", "Observed")})
for r in rows:
    for s in all_states:
        r.setdefault(s, "")
st.dataframe(
    pd.DataFrame(rows, columns=["Node", "Observed", *all_states]),
    hide_index=True,
    width="stretch",
)


# ---------------------------------------------------------------------------
# Full-width network diagram
# ---------------------------------------------------------------------------

st.markdown("<div class='section-title'>Network structure</div>", unsafe_allow_html=True)
fig = render_network(observed_nodes=list(evidence.keys()))
st.pyplot(fig, clear_figure=True)

st.caption(
    "Probabilities are illustrative — CPTs are expert-elicited, not "
    "calibrated from historical data. Translator output is an LLM "
    "reading of each headline and should be reviewed before acting on "
    "the resulting probabilities. See README for the BN-vs-HMM rationale."
)
