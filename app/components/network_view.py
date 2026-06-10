"""Network & model view: interactive DAG + node-detail posterior panel +
manual override (Plan 5 P4c). Extracted from the dashboard; its data deps are
injected as keyword args.
"""
from __future__ import annotations

from typing import Dict

from streamlit_agraph import agraph

# import the function (not the module): the override loop binds a local `state`,
# which would shadow a `state` module import.
from state import record_observation
from components import observed_node_panel
from components.ci_charts import (
    _ci_dataframe, _dumbbell_chart, _robustness_badge_html,
)
from src.network import STATES
from src.viz import TOPOLOGY_LAYOUT, build_agraph_payload
from theme import GREEN, MUTED, NAVY, RED, ROOT_DRIVER_STYLE


def render(st, *, all_marginals, evidence, soft_evidence, node_ci_table,
           observed_day_map, observed_meta, selected_bayes, topology):
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
            _topo_edges, _topo_levels = TOPOLOGY_LAYOUT[topology]
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
                    observed_node_panel.render(
                        st, observed_state=evidence[sel],
                        meta=observed_meta.get(sel, {}), bayes=selected_bayes,
                        marginal=marginal, sorted_states=sorted_states,
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
            if sel == "Scenario":
                st.markdown(
                    "<div class='card-sub'>The Scenario regime is inferred from the "
                    "evidence, not observed directly — no manual override.</div>",
                    unsafe_allow_html=True,
                )
            elif sel and sel in STATES:
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
                        record_observation(
                            headline=note.strip() or f"Manual: {sel} = {pinned}",
                            assignments={sel: pinned},
                            rationale="Set directly by the analyst via the network.",
                            per_assignment_reasons={sel: "Manual override."},
                            source="manual",
                        )
                    else:
                        dist = {s: v / 100.0 for s, v in vals.items()}
                        record_observation(
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
