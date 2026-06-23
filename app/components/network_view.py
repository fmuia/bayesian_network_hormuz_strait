"""Network & model view: interactive DAG + node-detail posterior panel +
manual override (Plan 5 P4c). Extracted from the dashboard; its data deps are
injected as keyword args.
"""
from __future__ import annotations

import io
from typing import Dict

import streamlit as _st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# import the function (not the module): the override loop binds a local `state`,
# which would shadow a `state` module import.
from state import override_to_observation, record_observation
from components import observed_node_panel
from components.ci_charts import (
    _ci_dataframe, _dumbbell_chart, _robustness_badge_html,
)
from src.scenario import LATENT, STATES
from src.viz import TOPOLOGY_LAYOUT, node_at_pixel, render_network_clickable
from theme import MUTED, ROOT_DRIVER_STYLE

# The diagram is displayed scaled to the page width (~900px, ~2x on retina), so a
# very high dpi is wasted bytes + render time. 110 stays crisp while roughly
# halving both the graphviz render and the PNG payload shipped to the browser.
_DAG_DPI = 110


@_st.cache_data(show_spinner=False, max_entries=128)
def _cached_dag(marginals, observed, observed_day, topology, selected):
    """Memoised DAG render. A click triggers two reruns (deliver the click, then
    apply the new selection); caching makes all but the one genuinely-new
    (state, selection) render a cache hit, so selecting a node is snappy."""
    edges, _ = TOPOLOGY_LAYOUT[topology]
    return render_network_clickable(
        marginals, observed=observed, observed_day=observed_day,
        edges=edges, selected=selected, dpi=_DAG_DPI,
    )


def _render_clickable_dag(st, *, all_marginals, evidence, observed_day_map,
                          topology):
    """Full-width tabular DAG (graphviz) with click-to-select node behaviour.

    Renders the same crisp diagram used in the docs, captures a click on the
    image, maps it back to a node via the graphviz geometry, and selects it —
    preserving the old agraph click-to-inspect/override flow on a far nicer
    image.
    """
    root_chip_html = "".join(
        (
            f"<span class='root-chip' style='background:{bg};"
            f"border:1px solid {border}; color:{border};'>"
            f"{node.replace('_', ' ')}</span>"
        )
        for node, (bg, border) in ROOT_DRIVER_STYLE.items()
    )
    st.markdown(
        "<div class='card-title'>Interactive DAG — click a node to inspect "
        "&amp; override</div>"
        "<div class='card-sub'>Each node card shows its posterior across states "
        "after propagating all injected evidence. Root drivers use dedicated "
        "color families; evidence-set nodes are filled. Click any node to load "
        "it into the Posterior &amp; Override panels below.</div>"
        f"<div>{root_chip_html}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)

    sel = st.session_state.selected_node
    png, boxes, png_w, _png_h = _cached_dag(
        all_marginals, evidence, observed_day_map, topology,
        sel if sel in STATES else None,
    )
    # The component needs a PIL image / path / ndarray (not raw bytes).
    coords = streamlit_image_coordinates(
        Image.open(io.BytesIO(png)),
        use_column_width="always", key="dag_click", cursor="pointer",
    )
    if coords is not None:
        # Streamlit replays the component's last value on every rerun; only act
        # on a genuinely new click (distinct timestamp).
        last = st.session_state.get("_dag_click_unix")
        if coords.get("unix_time") != last:
            st.session_state["_dag_click_unix"] = coords.get("unix_time")
            disp_w = coords.get("width") or png_w
            scale = png_w / disp_w  # displayed px → natural PNG px
            clicked = node_at_pixel(boxes, coords["x"] * scale, coords["y"] * scale)
            if clicked and clicked != st.session_state.selected_node:
                st.session_state.selected_node = clicked
                st.rerun()


def render(st, *, all_marginals, evidence, soft_evidence, node_ci_table,
           observed_day_map, observed_meta, selected_bayes, topology):
    # Full-width DAG on top; the Posterior + Override controls sit in a wide
    # row beneath it (each gets half the page, instead of a cramped sidebar).
    with st.container(border=True):
        _render_clickable_dag(
            st, all_marginals=all_marginals, evidence=evidence,
            observed_day_map=observed_day_map, topology=topology,
        )

    detail_col, override_col = st.columns(2, gap="large")

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
                        "CPT parameters are resampled (Dirichlet, per-CPT κ — "
                        "calibrated when an elicitation is locked, else 20; "
                        "m = 200)."
                    )
                else:
                    tip_text = (
                        "Posterior marginal after propagating all injected "
                        "evidence. Intervals come from CPT resampling "
                        "(Dirichlet, per-CPT κ — calibrated when an elicitation "
                        "is locked, else 20; m = 200)."
                    )
                # Node name with a native hover ⓘ tooltip (no dropdown/selector
                # chrome — the popover read as a control it isn't). Use the
                # `:gray[]` markdown directive, NOT unsafe_allow_html: the raw-HTML
                # path skips directive parsing and renders `help` as literal
                # ":help[]" text.
                st.markdown(
                    f":gray[**{sel.replace('_', ' ')}**]",
                    help=tip_text,
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
                    # Soft-observed node: show its standalone Bayes-factor
                    # contribution too (selected_bayes is None for unobserved).
                    if sel in soft_evidence:
                        observed_node_panel.render_bayes_contribution(st, selected_bayes)
            else:
                st.markdown(
                    "<div class='card-sub'>Click a node in the graph to inspect "
                    "its posterior distribution.</div>",
                    unsafe_allow_html=True,
                )

    with override_col:
        sel = st.session_state.selected_node
        with st.container(border=True):
            st.markdown(
                "<div class='card-title' style='margin-bottom:0.2rem;'>"
                "Override</div>",
                unsafe_allow_html=True,
            )
            if sel == LATENT:
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
                    # State name to the LEFT of the slider (not above) so the
                    # value bubble at 0 can't collide with the label.
                    lbl_col, sld_col = st.columns(
                        [1, 3], gap="small", vertical_alignment="center",
                    )
                    lbl_col.markdown(
                        f"<div class='slider-label'>{state}</div>",
                        unsafe_allow_html=True,
                    )
                    vals[state] = sld_col.slider(
                        state, 0, 100,
                        value=st.session_state.get(key, init),
                        step=1, key=key, label_visibility="collapsed",
                    )
                total = sum(vals.values())
                # Auto-normalise on apply (V4): the sliders no longer have to sum
                # to exactly 100. Preview what will actually be committed.
                pinned, dist = override_to_observation(vals)
                if total == 0:
                    preview = "Set at least one state above 0%."
                elif pinned is not None:
                    preview = f"Applies as a hard observation: <b>{pinned}</b>."
                else:
                    preview = "Applies as: " + " · ".join(
                        f"{s} {d * 100:.0f}%" for s, d in dist.items() if d > 0
                    )
                st.markdown(
                    f"<div style='font-size:0.82rem; margin-top:0.35rem; "
                    f"color:{MUTED};'>{preview}</div>",
                    unsafe_allow_html=True,
                )
                note = st.text_input(
                    "Note (optional)", key=f"note_{sel}",
                    placeholder="What drove this override?",
                )
                if st.button(
                    "Set observation",
                    type="primary",
                    disabled=(total == 0),
                    key=f"set_{sel}",
                ):
                    if pinned is not None:
                        record_observation(
                            headline=note.strip() or f"Manual: {sel} = {pinned}",
                            assignments={sel: pinned},
                            rationale="Set directly by the analyst via the network.",
                            per_assignment_reasons={sel: "Manual override."},
                            source="manual",
                        )
                    else:
                        pretty = ", ".join(
                            f"{s} {d * 100:.0f}%" for s, d in dist.items() if d > 0
                        )
                        record_observation(
                            headline=note.strip()
                                or f"Manual soft: {sel} ({pretty})",
                            assignments={},
                            soft_assignments={sel: dist},
                            rationale="Soft override set by the analyst (auto-normalised).",
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
