"""Streamlit dashboard for the Strait of Hormuz BN demo.

Run with: ``streamlit run app/dashboard.py`` (or ``pixi run app``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Make the project root importable when launched via `streamlit run`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.evidence import EVENTS, EvidenceEvent, events_by_category
from src.inference import BNInferenceEngine
from src.network import SCENARIO_NARRATIVES, STATES, build_network
from src.sensitivity import scenario_credible_intervals
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
      .block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; }}
      h1, h2, h3, h4 {{ color: {NAVY}; font-weight: 600; }}
      .demo-title {{
        font-size: 1.7rem; font-weight: 700; color: {NAVY};
        margin-bottom: 0.1rem;
      }}
      .demo-subtitle {{
        font-size: 0.95rem; color: #4B5563; margin-bottom: 1.2rem;
      }}
      .scenario-card {{
        background: {PANEL};
        border-left: 4px solid {NAVY};
        padding: 1rem 1.1rem;
        border-radius: 4px;
        height: 100%;
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
      .scenario-ci {{
        font-size: 0.8rem; color: #6B7280; margin-bottom: 0.55rem;
      }}
      .scenario-narrative {{
        font-size: 0.85rem; color: {NAVY}; line-height: 1.35;
      }}
      .section-title {{
        font-size: 1.0rem; font-weight: 600; color: {NAVY};
        margin: 1.2rem 0 0.4rem 0;
      }}
      .ev-count {{
        background: {TEAL}; color: white; padding: 0.15rem 0.5rem;
        border-radius: 10px; font-size: 0.75rem; font-weight: 600;
      }}
      [data-testid="stSidebar"] {{ background: white; }}
      div[data-testid="stExpander"] details {{
        border: 1px solid #E5E7EB; border-radius: 4px;
      }}
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
    """Cache CI computation keyed on a hashable evidence representation."""
    return scenario_credible_intervals(dict(evidence_items), m=200, concentration=20.0)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "active_event_ids" not in st.session_state:
    st.session_state.active_event_ids = []  # ordered list of toggled event ids
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, float]] = []
if "history_labels" not in st.session_state:
    st.session_state.history_labels: List[str] = []


def _evidence_from_active(active_ids: List[str]) -> Dict[str, str]:
    """Merge assignments from toggled events; later events override earlier."""
    by_id = {e.id: e for e in EVENTS}
    merged: Dict[str, str] = {}
    for eid in active_ids:
        merged.update(by_id[eid].assignments)
    return merged


# ---------------------------------------------------------------------------
# Sidebar — evidence controls
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Evidence")
col_a, col_b = st.sidebar.columns([1, 1])
with col_a:
    if st.button("Clear all", width="stretch"):
        st.session_state.active_event_ids = []
        st.session_state.history = []
        st.session_state.history_labels = []
        st.rerun()
with col_b:
    st.markdown(
        f"<div style='text-align:right; padding-top:0.45rem;'>"
        f"<span class='ev-count'>{len(st.session_state.active_event_ids)} active</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

prev_active = list(st.session_state.active_event_ids)
new_active: List[str] = []
grouped = events_by_category()
category_order = ["de-escalation", "mixed", "escalation"]
category_label = {
    "de-escalation": "De-escalation",
    "mixed": "Mixed signals",
    "escalation": "Escalation",
}
for cat in category_order:
    with st.sidebar.expander(category_label[cat], expanded=(cat == "mixed")):
        for ev in grouped[cat]:
            checked = st.checkbox(
                f"**{ev.date}** — {ev.headline}",
                value=ev.id in prev_active,
                key=f"chk_{ev.id}",
            )
            if checked:
                new_active.append(ev.id)

# Detect change vs previous active set; track history of *added* events.
if new_active != prev_active:
    added = [e for e in new_active if e not in prev_active]
    st.session_state.active_event_ids = new_active
    if added:
        # Compute scenario probs after this update and append to history.
        engine = get_engine()
        engine.clear_evidence()
        engine.update_evidence(_evidence_from_active(new_active))
        probs = engine.get_scenario_probabilities()
        by_id = {e.id: e for e in EVENTS}
        for added_id in added:
            st.session_state.history.append(probs)
            st.session_state.history_labels.append(by_id[added_id].headline)
    else:
        # Recompute history snapshot for removal so the chart stays sensible.
        engine = get_engine()
        engine.clear_evidence()
        engine.update_evidence(_evidence_from_active(new_active))
        probs = engine.get_scenario_probabilities()
        st.session_state.history.append(probs)
        st.session_state.history_labels.append("(evidence removed)")

# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

engine = get_engine()
engine.clear_evidence()
evidence = _evidence_from_active(st.session_state.active_event_ids)
engine.update_evidence(evidence)

scenario_probs = engine.get_scenario_probabilities()
with st.spinner("Quantifying parameter uncertainty (200 Monte-Carlo runs)…"):
    ci_table = cached_credible_intervals(tuple(sorted(evidence.items())))

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    "<div class='demo-title'>Adaptive Scenario Probability Framework "
    "&mdash; Strait of Hormuz Demo</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='demo-subtitle'>Scenario probabilities update as new "
    "evidence is added on the left. Bands show 80% credible intervals "
    "from second-order parameter uncertainty.</div>",
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
# History chart + network viz
# ---------------------------------------------------------------------------

st.markdown("<div class='section-title'>Probability evolution this session</div>",
            unsafe_allow_html=True)
hist_col, net_col = st.columns([1.1, 1.0])

with hist_col:
    if not st.session_state.history:
        st.info("Toggle evidence in the sidebar to start tracking probability changes.")
    else:
        df = pd.DataFrame(st.session_state.history)
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        for scenario in ["Stress_Mitigates", "Prolonged_Conflict", "Severe_Closure"]:
            ax.plot(
                range(1, len(df) + 1), df[scenario],
                marker="o", linewidth=2.2, markersize=5,
                color=SCENARIO_COLOR[scenario], label=SCENARIO_LABEL[scenario],
            )
        ax.set_xlabel("Evidence update #", fontsize=9, color=NAVY)
        ax.set_ylabel("Probability", fontsize=9, color=NAVY)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors=NAVY, labelsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

with net_col:
    fig = render_network(observed_nodes=list(evidence.keys()))
    st.pyplot(fig, clear_figure=True)
    st.caption("Filled teal: nodes constrained by current evidence. Navy: terminal scenario node.")

# ---------------------------------------------------------------------------
# Intermediate node marginals
# ---------------------------------------------------------------------------

st.markdown("<div class='section-title'>Intermediate node marginals given current evidence</div>",
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

# Pad rows so all share the same columns, leaving blanks for nodes with
# fewer states. (All intermediate nodes here have 3 states; this is
# robust to future edits.)
all_states = sorted({k for r in rows for k in r if k not in ("Node", "Observed")})
for r in rows:
    for s in all_states:
        r.setdefault(s, "")

st.dataframe(
    pd.DataFrame(rows, columns=["Node", "Observed", *all_states]),
    hide_index=True,
    width="stretch",
)

st.caption(
    "Probabilities are illustrative — CPTs are expert-elicited, not "
    "calibrated from historical data. See README for the BN-vs-HMM rationale."
)
