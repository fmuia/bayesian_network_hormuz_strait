"""Probability-evolution-by-day chart (Plan 5 P3). Extracted from app/dashboard.py.

The engine + the cached-CI function are injected so the component stays free of
the dashboard's caching/engine wiring (A3 will replace the mutation pattern).
"""
from __future__ import annotations

from typing import Dict, List

import altair as alt
import pandas as pd

from theme import AMBER, GREEN, MUTED, NAVY, RED, SCENARIO_LABEL

# The evolution chart is the locus of attention, so it gets the reallocated space
# from the compacted scenario cards (Plan 5 P7 / C1 / V2) — ≥1.5× the old 260px.
_CHART_HEIGHT = 400


def render_evolution_chart(st, observations, *, engine, cached_ci,
                           locked_spec_json, topology):
    with st.container(border=True):
        st.markdown("<div class='card-title'>Probability evolution by day</div>",
                    unsafe_allow_html=True)

        if not observations:
            st.markdown(
                f"<div class='card-sub' style='color:{MUTED};'>"
                "No observations yet — the timeline fills as you translate or "
                "override observations.</div>",
                unsafe_allow_html=True,
            )
        else:
            history_rows: List[Dict] = []
            engine_h = engine
            engine_h.clear_evidence()
            priors = engine_h.get_prior_probabilities()
            prior_ci = cached_ci(tuple(), topology, locked_spec_json)
            history_rows.append({
                "Day": 0, "HeadlinesOnDay": "(prior)", "n_obs": 0,
                "ci": prior_ci, **priors,
            })

            grouped: Dict[int, List[Dict]] = {}
            for obs in observations:
                grouped.setdefault(obs["day"], []).append(obs)

            cum_hard: Dict[str, str] = {}
            cum_soft: Dict[str, Dict[str, float]] = {}
            with st.spinner("Quantifying parameter uncertainty per day…"):
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
                    day_ci_evidence = dict(cum_hard)
                    for node, dist in cum_soft.items():
                        day_ci_evidence[node] = max(dist, key=dist.get)
                    day_ci = cached_ci(
                        tuple(sorted(day_ci_evidence.items())), topology, locked_spec_json
                    )
                    history_rows.append({
                        "Day": day,
                        "HeadlinesOnDay": headlines,
                        "n_obs": len(day_obs),
                        "ci": day_ci,
                        **engine_h.get_scenario_probabilities(),
                    })

            wide = pd.DataFrame(history_rows)
            long_rows = []
            for _, r in wide.iterrows():
                for sc in ["Stress_Mitigates", "Prolonged_Conflict", "Severe_Closure"]:
                    mean_ci, lo_ci, hi_ci = r["ci"][sc]
                    long_rows.append({
                        "Day": int(r["Day"]),
                        "Scenario": SCENARIO_LABEL[sc],
                        "ScenarioKey": sc,
                        # Use the Dirichlet-resample mean so the line is the
                        # centre of the CI band (and matches the scenario
                        # cards). The unperturbed posterior from
                        # get_scenario_probabilities() can lie outside
                        # [Lo, Hi] because inference is non-linear in the CPTs.
                        "Probability": float(mean_ci),
                        "Lo": float(lo_ci),
                        "Hi": float(hi_ci),
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
            bands = base.mark_area(opacity=0.18, interpolate="linear").encode(
                y=alt.Y("Lo:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
                y2="Hi:Q",
            )
            lines = base.mark_line(strokeWidth=2.6, point=alt.OverlayMarkDef(size=70))
            hover = alt.selection_point(
                fields=["Day"], nearest=True, on="mouseover", empty=False,
            )
            tooltip = base.mark_circle(size=120, opacity=0).encode(
                tooltip=[
                    alt.Tooltip("Day:O"),
                    alt.Tooltip("Scenario:N"),
                    alt.Tooltip("Probability:Q", format=".1%", title="Mean"),
                    alt.Tooltip("Lo:Q", format=".1%", title="Lo (10%)"),
                    alt.Tooltip("Hi:Q", format=".1%", title="Hi (90%)"),
                    alt.Tooltip("n_obs:Q", title="# obs added"),
                    alt.Tooltip("HeadlinesOnDay:N", title="Headlines"),
                ],
            ).add_params(hover)
            chart = (bands + lines + tooltip).properties(
                height=_CHART_HEIGHT,
            ).configure_view(stroke=None)
            st.altair_chart(chart, width="stretch")
            st.caption(
                "Lines are the Dirichlet-resample posterior mean (matching the "
                "scenario cards above). Shaded bands are the 80% credible "
                "interval from CPT resampling at each day's evidence state "
                "(concentration = 20, m = 200). They reflect parameter "
                "uncertainty **at that day**, not forecast uncertainty about "
                "future trajectories."
            )

            last_day = max(grouped)
            last_headlines = " · ".join(o["headline"] for o in grouped[last_day])
            st.caption(
                f"Most recent update — Day {last_day}: {last_headlines}"
            )


    # CI / robustness chart helpers live in app/components/ci_charts.py (Plan 5 P3).
    from components.ci_charts import (  # noqa: E402
        _ci_dataframe,
        _dumbbell_chart,
        _flat_bar_chart,
        _robustness_badge_html,
        _width_category,
    )
