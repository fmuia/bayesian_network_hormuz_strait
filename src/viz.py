"""Tabular BN diagram rendered with graphviz.

Each node becomes a small HTML-label table showing either (a) the
observed state plus the day it was recorded, or (b) the posterior
marginal across that node's states given current evidence. Edges use
the BN's DAG structure.

The renderer returns raw PNG bytes so Streamlit can display them via
``st.image`` without an intermediate file.
"""

from __future__ import annotations

import subprocess
from html import escape
from typing import Dict, Iterable, Mapping, Optional

import graphviz

from .network import EDGES, STATES

_PLUGINS_REGISTERED = False


def _ensure_plugins_registered() -> None:
    """Run ``dot -c`` once per process.

    Conda-forge's graphviz package sometimes ships without a populated
    plugin config, so the first ``dot`` call fails with 'no layout
    engine support for "dot"'. Running ``dot -c`` regenerates the
    config; this is idempotent and cheap.
    """
    global _PLUGINS_REGISTERED
    if _PLUGINS_REGISTERED:
        return
    try:
        subprocess.run(["dot", "-c"], check=False, capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    _PLUGINS_REGISTERED = True

# Palette (must stay in sync with the dashboard).
_TEAL = "#1A7A6D"
_TEAL_DARK = "#14635A"
_NAVY = "#1B2A3D"
_PANEL = "#F5F5F5"
_BORDER = "#CBD5E1"
_BAR_BG = "#E5E7EB"
_SCENARIO_COLORS = {
    "Stress_Mitigates": "#2E8B57",
    "Prolonged_Conflict": "#D4A017",
    "Severe_Closure": "#B22222",
}


def _bar(pct: float, fill: str, width: int = 80) -> str:
    """Horizontal proportional bar rendered as two nested 1-cell tables."""
    pct = max(0.0, min(100.0, pct))
    filled_w = max(1, int(round(width * pct / 100)))
    empty_w = max(1, width - filled_w)
    return (
        f'<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        f'<TR>'
        f'<TD BGCOLOR="{fill}" WIDTH="{filled_w}" HEIGHT="8"></TD>'
        f'<TD BGCOLOR="{_BAR_BG}" WIDTH="{empty_w}" HEIGHT="8"></TD>'
        f'</TR></TABLE>'
    )


def _prob_rows(marginal: Mapping[str, float], fill: str) -> str:
    """Build the state/prob/bar rows for an unobserved node."""
    rows = []
    for state, prob in marginal.items():
        pct = prob * 100
        rows.append(
            '<TR>'
            f'<TD ALIGN="LEFT"><FONT POINT-SIZE="9" COLOR="{_NAVY}">'
            f'{escape(state)}</FONT></TD>'
            f'<TD ALIGN="RIGHT"><FONT POINT-SIZE="9" COLOR="{_NAVY}">'
            f'{pct:4.1f}%</FONT></TD>'
            f'<TD>{_bar(pct, fill)}</TD>'
            '</TR>'
        )
    return "".join(rows)


def _node_label(
    node: str,
    marginal: Mapping[str, float],
    observed_state: Optional[str],
    day: Optional[int],
) -> str:
    """Return a Graphviz HTML-like label for one node."""
    title = escape(node.replace("_", " "))

    if node == "Scenario":
        # Scenario terminal: show all three probs colour-coded per scenario.
        rows = []
        for state, prob in marginal.items():
            pct = prob * 100
            color = _SCENARIO_COLORS.get(state, _NAVY)
            rows.append(
                '<TR>'
                f'<TD ALIGN="LEFT"><FONT POINT-SIZE="9" COLOR="white"><B>'
                f'{escape(state.replace("_", " "))}</B></FONT></TD>'
                f'<TD ALIGN="RIGHT"><FONT POINT-SIZE="9" COLOR="white"><B>'
                f'{pct:4.1f}%</B></FONT></TD>'
                f'<TD>{_bar(pct, color)}</TD>'
                '</TR>'
            )
        header = (
            f'<TR><TD COLSPAN="3" BGCOLOR="{_NAVY}" ALIGN="CENTER">'
            f'<FONT POINT-SIZE="11" COLOR="white"><B>SCENARIO</B></FONT>'
            '</TD></TR>'
        )
        return (
            f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" '
            f'CELLPADDING="4" BGCOLOR="{_NAVY}">'
            f'{header}{"".join(rows)}</TABLE>>'
        )

    if observed_state is not None:
        # Observed-node card: teal fill, shows the observed state + day.
        day_badge = (
            f'<TD ALIGN="RIGHT"><FONT POINT-SIZE="8" COLOR="#D5EFEA">'
            f'Day {day}</FONT></TD>'
        ) if day is not None else '<TD></TD>'
        return (
            f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" '
            f'CELLPADDING="5" BGCOLOR="{_TEAL}">'
            f'<TR>'
            f'<TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="white"><B>'
            f'{title}</B></FONT></TD>'
            f'{day_badge}'
            f'</TR>'
            f'<TR><TD COLSPAN="2" BGCOLOR="{_TEAL_DARK}" ALIGN="CENTER">'
            f'<FONT POINT-SIZE="11" COLOR="white"><B>'
            f'● {escape(observed_state)}</B></FONT></TD></TR>'
            f'</TABLE>>'
        )

    # Default card: probability distribution.
    header = (
        f'<TR><TD COLSPAN="3" BGCOLOR="{_PANEL}" ALIGN="LEFT">'
        f'<FONT POINT-SIZE="10" COLOR="{_NAVY}"><B>{title}</B></FONT>'
        '</TD></TR>'
    )
    rows = _prob_rows(marginal, fill=_TEAL)
    return (
        f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" '
        f'CELLPADDING="4" BGCOLOR="white">'
        f'{header}{rows}</TABLE>>'
    )


def render_network_png(
    marginals: Mapping[str, Mapping[str, float]],
    *,
    observed: Mapping[str, str] = {},
    observed_day: Mapping[str, int] = {},
    dpi: int = 220,
) -> bytes:
    """Render the BN as PNG bytes for display in Streamlit.

    ``marginals`` must contain every node; ``observed`` maps evidence
    nodes to their set state; ``observed_day`` maps those same nodes
    to the day-of-session they were first set.
    """
    _ensure_plugins_registered()
    dot = graphviz.Digraph(
        "bayesian_network",
        graph_attr={
            "rankdir": "LR",
            "bgcolor": "white",
            "splines": "spline",
            "nodesep": "0.35",
            "ranksep": "0.9",
            "fontname": "Helvetica",
            "dpi": str(dpi),
        },
        node_attr={
            "shape": "plain",
            "fontname": "Helvetica",
            "margin": "0",
        },
        edge_attr={
            "color": "#94A3B8",
            "arrowsize": "0.7",
            "penwidth": "1.1",
        },
    )

    for node in STATES.keys():
        label = _node_label(
            node=node,
            marginal=marginals[node],
            observed_state=observed.get(node),
            day=observed_day.get(node),
        )
        # When a node is observed, draw a subtle border around it too.
        node_attrs = {"label": label}
        dot.node(node, **node_attrs)

    for src, dst in EDGES:
        style = "solid"
        color = "#94A3B8"
        if src in observed:
            color = _TEAL
        dot.edge(src, dst, style=style, color=color)

    return dot.pipe(format="png")


# ---------------------------------------------------------------------------
# Interactive (streamlit-agraph) payload
# ---------------------------------------------------------------------------

# Hierarchical "levels" for the vis.js hierarchical layout — roots on the
# left, scenario on the right. Chosen manually for a stable left-to-right
# DAG layout that does not jitter between reruns.
_ROOT_DRIVER_COLORS: Dict[str, tuple] = {
    # (background, border, observed_fill) — distinct hues per root driver.
    "US_Iran_Negotiations": ("#DBEAFE", "#1D4ED8", "#1E40AF"),  # blue
    "Iranian_Regime_Stability": ("#FCE7F3", "#BE185D", "#9D174D"),  # rose
    "Third_Party_Mediation": ("#FEF3C7", "#B45309", "#92400E"),  # amber
    "Sanctions_Trajectory": ("#EDE9FE", "#6D28D9", "#5B21B6"),  # violet
}

_NODE_LEVEL: Dict[str, int] = {
    "US_Iran_Negotiations": 0,
    "Iranian_Regime_Stability": 0,
    "Third_Party_Mediation": 0,
    "Sanctions_Trajectory": 0,
    "Iranian_Proxy_Activity": 1,
    "Tanker_Incidents": 1,
    "US_Military_Response": 2,
    "Strait_Operationally_Closed": 2,
    "Energy_Infrastructure_Damage": 3,
    "Conflict_Duration": 3,
    "Diplomatic_Resolution_Path": 3,
    "Oil_Price_Regime": 3,
    "Scenario": 4,
}


def _display_name(raw: str) -> str:
    return raw.replace("_", " ")


def _wrap_node_title(name: str) -> str:
    """Wrap long node titles across two lines for readability."""
    replacements = {
        "US Iran Negotiations": "US Iran\nNegotiations",
        "Iranian Regime Stability": "Iranian Regime\nStability",
        "Third Party Mediation": "Third Party\nMediation",
        "Sanctions Trajectory": "Sanctions\nTrajectory",
        "Iranian Proxy Activity": "Iranian Proxy\nActivity",
        "Tanker Incidents": "Tanker\nIncidents",
        "US Military Response": "US Military\nResponse",
        "Strait Operationally Closed": "Strait Operationally\nClosed",
        "Energy Infrastructure Damage": "Energy Infrastructure\nDamage",
        "Conflict Duration": "Conflict\nDuration",
        "Diplomatic Resolution Path": "Diplomatic Resolution\nPath",
        "Oil Price Regime": "Oil Price\nRegime",
    }
    return replacements.get(name, name)


def _format_label_text(
    node: str,
    marginal: Mapping[str, float],
    observed_state: Optional[str],
    day: Optional[int],
) -> str:
    """Build compact, readable node label to avoid overlap."""
    title = _wrap_node_title(_display_name(node))
    lines = [title]
    if observed_state is not None:
        day_suffix = f" · day {day}" if day is not None else ""
        lines.append(f"Observed: {_display_name(observed_state)}{day_suffix}")
    else:
        top_state = max(STATES[node], key=lambda s: marginal.get(s, 0.0))
        top_prob = marginal.get(top_state, 0.0)
        lines.append(f"{_display_name(top_state)}  {top_prob*100:0.1f}%")
    return "\n".join(lines)


def _format_tooltip_text(
    node: str,
    marginal: Mapping[str, float],
    observed_state: Optional[str],
    day: Optional[int],
) -> str:
    """Build hover tooltip with full probabilities in plain text."""
    rows = [_display_name(node)]
    if observed_state is not None:
        day_suffix = f" (day {day})" if day is not None else ""
        rows.append(f"Observed: {_display_name(observed_state)}{day_suffix}")
    for state in STATES[node]:
        prob = marginal.get(state, 0.0)
        rows.append(f"{_display_name(state)}: {prob*100:0.1f}%")
    return "\n".join(rows)


def build_agraph_payload(
    marginals: Mapping[str, Mapping[str, float]],
    *,
    observed: Mapping[str, str] = {},
    observed_day: Mapping[str, int] = {},
):
    """Return ``(nodes, edges, config)`` for ``streamlit_agraph.agraph``.

    The dashboard uses this for the interactive network view: nodes are
    clickable (returns the clicked node id), and vis.js provides pan /
    zoom / hover natively.
    """
    # Lazy-import so the rest of the package keeps working without the dep.
    from streamlit_agraph import Config, Edge, Node

    nodes = []
    for node in STATES.keys():
        marginal = marginals[node]
        obs_state = observed.get(node)
        day = observed_day.get(node)
        is_root_driver = node in _ROOT_DRIVER_COLORS

        if node == "Scenario":
            label = _format_label_text(node, marginal, obs_state, day)
            color = _NAVY
            font_color = "white"
            border_color = _NAVY
            size = 34
        elif obs_state is not None:
            label = _format_label_text(node, marginal, obs_state, day)
            if is_root_driver:
                _, border_color, observed_fill = _ROOT_DRIVER_COLORS[node]
                color = observed_fill
                font_color = "white"
                size = 30
            else:
                color = _TEAL
                font_color = "white"
                border_color = _TEAL_DARK
                size = 28
        else:
            label = _format_label_text(node, marginal, obs_state, day)
            if is_root_driver:
                color, border_color, _ = _ROOT_DRIVER_COLORS[node]
            else:
                color = "white"
                border_color = _BORDER
            font_color = _NAVY
            size = 26 if is_root_driver else 24

        nodes.append(
            Node(
                id=node,
                label=label,
                title=_format_tooltip_text(node, marginal, obs_state, day),
                size=size,
                shape="box",
                color={
                    "background": color,
                    "border": border_color,
                    "highlight": {"background": color, "border": _TEAL_DARK},
                },
                font={
                    "color": font_color,
                    "size": 12,
                    "face": "Inter, Helvetica, Arial",
                },
                borderWidth=3 if is_root_driver else (2 if obs_state is not None else 1),
                level=_NODE_LEVEL.get(node, 0),
                margin=4,
            )
        )

    edges = []
    for src, dst in EDGES:
        highlight = src in observed or dst in observed
        src_root_style = _ROOT_DRIVER_COLORS.get(src)
        edge_color = src_root_style[1] if src_root_style is not None else "#94A3B8"
        edge_width = 1.8 if src_root_style is not None else 1
        if highlight:
            edge_width = max(edge_width, 2.2)
        edges.append(
            Edge(
                source=src,
                target=dst,
                color=edge_color,
                width=edge_width,
            )
        )

    config = Config(
        width="100%",
        height=520,
        directed=True,
        physics=False,
        hierarchical=True,
        nodeHighlightBehavior=True,
        highlightColor=_TEAL,
        collapsible=False,
        node={"labelProperty": "label", "renderLabel": True},
        link={"renderLabel": False},
        # vis.js nested hierarchical options
        layout={
            "hierarchical": {
                "enabled": True,
                "direction": "LR",
                "sortMethod": "directed",
                "levelSeparation": 180,
                "nodeSpacing": 175,
                "treeSpacing": 175,
            }
        },
        interaction={"hover": True, "zoomView": False, "dragView": False},
        manipulation=False,
    )

    return nodes, edges, config


# Backwards-compat shim for any caller still using the old matplotlib
# entry point. Returns a Figure-less None; the dashboard no longer uses it.
def render_network(observed_nodes: Iterable[str] = ()):  # pragma: no cover
    raise RuntimeError(
        "render_network() has been replaced by render_network_png(); "
        "update your caller to pass marginals."
    )


__all__ = ["render_network_png", "build_agraph_payload"]
