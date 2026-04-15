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


# Backwards-compat shim for any caller still using the old matplotlib
# entry point. Returns a Figure-less None; the dashboard no longer uses it.
def render_network(observed_nodes: Iterable[str] = ()):  # pragma: no cover
    raise RuntimeError(
        "render_network() has been replaced by render_network_png(); "
        "update your caller to pass marginals."
    )


__all__ = ["render_network_png"]
