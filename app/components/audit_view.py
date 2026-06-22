"""Audit-trail view (Plan 5 P4). Extracted from the dashboard."""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from src.scenario import PRESENTATION, STATES


def render(st, *, engine, evidence, soft_evidence):
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

    intermediate_nodes = PRESENTATION.intermediate_nodes

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
