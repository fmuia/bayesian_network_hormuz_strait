"""Pinned scenario-outlook cards + the uncertainty-detail expander (Plan 5 P3 / C1).

Extracted verbatim from app/dashboard.py; consumes the credible-interval table.
"""
from __future__ import annotations

import altair as alt
import pandas as pd

from src.network import SCENARIO_NARRATIVES
from theme import AMBER, GREEN, NAVY, RED, SCENARIO_COLOR, SCENARIO_LABEL


def render_scenario_outlook(st, ci_table):
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
            st.altair_chart(err_chart, width="stretch")
            st.caption(
                "Intervals come from resampling CPT parameters (Dirichlet, "
                "concentration = 20, m = 200) and re-running inference."
            )
