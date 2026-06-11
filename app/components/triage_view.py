"""Triage / HITL review-queue view (Plan 5 P4). Extracted from the dashboard."""
from __future__ import annotations

import state
from src.network import STATES


def render(st):
    st.markdown(
        "<div class='card-title'>Triage — translations awaiting review</div>"
        "<div class='card-sub'>Flagged translations (partial relevance, or all of "
        "them when “Require review before inject” is on) wait here and do "
        "<b>not</b> affect the model until you act. Approve, edit a state, or "
        "reject.</div>",
        unsafe_allow_html=True,
    )
    _queue = st.session_state.review_queue
    if not _queue:
        st.info(
            "Nothing awaiting review. Clearly-relevant translations auto-inject; "
            "off-topic ones abstain. Turn on “Require review before inject” in the "
            "sidebar to route everything here first."
        )
    for _item in list(_queue):
        with st.container(border=True):
            _rel = _item.get("relevance", "yes")
            _badge = (
                " <span class='assign-chip chip-warn'>⚠ partial</span>"
                if _rel == "partial" else ""
            )
            st.markdown(
                f"<div class='translator-headline'>“{_item['headline']}”{_badge}</div>"
                f"<div class='meta'>day {_item['day']} · {_item['provider']} · "
                f"{_item['model']}</div>",
                unsafe_allow_html=True,
            )
            chips = "".join(
                f"<span class='assign-chip'>{a['node'].replace('_',' ')} = {a['state']}</span>"
                for a in _item["assignments"]
            )
            st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)

            a_col, r_col = st.columns(2, gap="small")
            if a_col.button("✓ Approve", key=f"appr_{_item['id']}", width="stretch",
                            type="primary"):
                state.inject_review_item(_item)
                state.remove_review_item(_item["id"])
                st.rerun()
            if r_col.button("✕ Reject", key=f"rej_{_item['id']}", width="stretch"):
                state.remove_review_item(_item["id"])
                st.rerun()

            with st.expander("Edit states before approving"):
                _overrides = {}
                for a in _item["assignments"]:
                    node = a["node"]
                    _overrides[node] = st.selectbox(
                        node.replace("_", " "),
                        STATES[node],
                        index=STATES[node].index(a["state"]),
                        key=f"edit_{_item['id']}_{node}",
                    )
                if st.button("✓ Approve with edits", key=f"appredit_{_item['id']}",
                             width="stretch"):
                    state.inject_review_item(_item, state_overrides=_overrides)
                    state.remove_review_item(_item["id"])
                    st.rerun()
