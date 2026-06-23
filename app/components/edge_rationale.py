"""Edge-rationale view (Plan 5 P4). The per-topology edge rationale + omission
tables are scenario-specific and come from the active pack (``PRESENTATION``);
the rendering logic here is generic.
"""
from __future__ import annotations

from src.scenario import NODE_META, PRESENTATION


def _fmt_node(name: str) -> str:
    meta = NODE_META.get(name)
    return meta.label if meta else name.replace("_", " ")


def edge_title_map(topology) -> dict:
    """``{(parent, child): rationale}`` for the *present* edges of a topology —
    used to populate the agraph edge hover tooltips (Plan 5 P10 / C11 / V13).
    Omitted edges are not in the graph, so they are not included.
    """
    rationale = PRESENTATION.edge_rationales.get(topology, [])
    return {(parent, child): reason for parent, child, reason in rationale}


def render(st, topology):
    st.markdown(
        "<div class='card-title'>Why each arrow is (or isn't) in the "
        "network</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Short rationales for every edge in the DAG, plus a list of "
        "plausible connections that were deliberately left out and why. "
        "These are modeling judgments, not settled truths — treat them "
        "as the assumptions behind the current CPTs."
    )

    _rationale = PRESENTATION.edge_rationales.get(topology, [])
    _omissions = PRESENTATION.edge_omissions.get(topology, [])

    st.markdown("#### Edges present in the model")
    for parent, child, reason in _rationale:
        st.markdown(
            f"**{_fmt_node(parent)}** → **{_fmt_node(child)}**  \n"
            f"<span style='color:#475569'>{reason}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Notable omitted edges")
    st.caption(
        "Connections a domain reader might expect but that are not in "
        "the DAG — either because they're mediated by another node or "
        "because they were dropped for parsimony."
    )
    for parent, child, reason in _omissions:
        st.markdown(
            f"**{_fmt_node(parent)}** ⇢ **{_fmt_node(child)}** "
            f"<span style='color:#b45309'>(omitted)</span>  \n"
            f"<span style='color:#475569'>{reason}</span>",
            unsafe_allow_html=True,
        )
