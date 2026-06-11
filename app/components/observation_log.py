"""Observations view: latest translation + day-grouped observation log
(Plan 5 P4). Extracted from the dashboard."""
from __future__ import annotations

import html
from typing import Dict, List

from theme import MUTED
from components import structured_panel


def render(st):
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
            ) or "<span class='muted-note'>No assignments</span>"
            # B3 relevance badge.
            _rel = t.get("relevance", "yes")
            _rel_badge = {
                "no": "<span class='assign-chip chip-danger'>"
                      "⛔ not relevant — not injected</span>",
                "partial": "<span class='assign-chip chip-warn'>"
                           "⚠ partial relevance — review before relying on it</span>",
            }.get(_rel, "")
            st.markdown(
                f"""
                <div class='translator-headline'>“{html.escape(t['headline'])}”</div>
                <div class='translator-rationale'>{html.escape(t['rationale'])}</div>
                <div>{_rel_badge}{chips_html}</div>
                <div class='meta'>provider: {t.get('provider','?')} ·
                model: {t['model']} · relevance: {_rel}</div>
                """,
                unsafe_allow_html=True,
            )
            if t.get("pending_review"):
                st.warning(
                    "⏳ **Pending review — not yet injected.** Approve, edit a state, "
                    "or reject it in the **⚖️ Triage** view; until then it does not "
                    "affect the model."
                )
            structured_panel.render(st, t)
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
                    _src = (obs.get("source") or "").strip()
                    _src_html = (
                        f" <span style='color:{MUTED}; font-size:0.72rem;'>"
                        f"({html.escape(_src)})</span>" if _src else ""
                    )
                    row_col, btn_col = st.columns([20, 1])
                    with row_col:
                        st.markdown(
                            f"<div class='obs-row{first_cls}'>"
                            f"<div class='obs-headline'>{html.escape(obs['headline'])}{_src_html}</div>"
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
