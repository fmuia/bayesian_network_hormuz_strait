"""Tabular BN diagram rendered with graphviz.

Each node becomes a small HTML-label table showing either (a) the
observed state plus the day it was recorded, or (b) the posterior
marginal across that node's states given current evidence. Edges use
the BN's DAG structure.

The renderer returns raw PNG bytes so Streamlit can display them via
``st.image`` without an intermediate file.
"""

from __future__ import annotations

import json
import struct
import subprocess
from html import escape
from typing import Dict, Iterable, Mapping, Optional, Tuple

import graphviz

from src.scenario import (
    DISPLAY_OVERRIDES,
    EDGES,
    EDGES_LATENT,
    LATENT,
    LAYOUT as TOPOLOGY_LAYOUT,
    NODE_TITLE_WRAP,
    PRESENTATION,
    STATES,
)

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
_SELECT = "#14B8A6"  # selection border — bright teal, reads on white & navy
_NODE_BORDER = "black"  # default border on every node card
_SCENARIO_COLORS = PRESENTATION.scenario_color
# (bg_light, border_dark, observed_fill) per root driver — shared by the static
# graphviz renderer and the agraph payload below.
_ROOT_DRIVER_COLORS: Dict[str, tuple] = PRESENTATION.root_driver_colors


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


def _table(inner: str, *, bg: str, cellpadding: int, selected: bool) -> str:
    """Wrap node rows in the outer HTML-label table.

    ``selected`` draws a thick accent border so the clicked node is obvious
    in the otherwise-static image.
    """
    # Every card carries a border: black by default, bright-green and thicker
    # when the node is selected.
    border, color = ("4", _SELECT) if selected else ("2", _NODE_BORDER)
    return (
        f"<<TABLE BORDER=\"{border}\" COLOR=\"{color}\" CELLBORDER=\"0\" "
        f"CELLSPACING=\"0\" CELLPADDING=\"{cellpadding}\" BGCOLOR=\"{bg}\">"
        f"{inner}</TABLE>>"
    )


def _node_label(
    node: str,
    marginal: Mapping[str, float],
    observed_state: Optional[str],
    day: Optional[int],
    *,
    root_style: Optional[Tuple[str, str, str]] = None,
    selected: bool = False,
) -> str:
    """Return a Graphviz HTML-like label for one node.

    ``root_style`` is the ``(bg_light, border_dark, observed_fill)`` colour
    family for a root driver (``None`` for every other node), so the static
    diagram matches the interactive view's dedicated driver colours.
    """
    title = escape(node.replace("_", " "))

    if node == LATENT:
        # Scenario terminal: show all states colour-coded per scenario.
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
            f'<FONT POINT-SIZE="11" COLOR="white"><B>{title.upper()}</B></FONT>'
            '</TD></TR>'
        )
        return _table(header + "".join(rows), bg=_NAVY, cellpadding=4, selected=selected)

    if observed_state is not None:
        # Observed-node card: filled, shows the observed state + day. Root
        # drivers keep their own colour family; everything else is teal.
        fill = root_style[2] if root_style is not None else _TEAL
        subbar = root_style[1] if root_style is not None else _TEAL_DARK
        day_badge = (
            f'<TD ALIGN="RIGHT"><FONT POINT-SIZE="8" COLOR="#E8F0FE">'
            f'Day {day}</FONT></TD>'
        ) if day is not None else '<TD></TD>'
        inner = (
            f'<TR>'
            f'<TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="white"><B>'
            f'{title}</B></FONT></TD>'
            f'{day_badge}'
            f'</TR>'
            f'<TR><TD COLSPAN="2" BGCOLOR="{subbar}" ALIGN="CENTER">'
            f'<FONT POINT-SIZE="11" COLOR="white"><B>'
            f'● {escape(observed_state)}</B></FONT></TD></TR>'
        )
        return _table(inner, bg=fill, cellpadding=5, selected=selected)

    # Default card: probability distribution. Root drivers tint the header and
    # bars with their colour family; other nodes use the neutral panel + teal.
    if root_style is not None:
        bg_light, border_dark, _ = root_style
        header_bg, title_color, bar_fill = bg_light, border_dark, border_dark
    else:
        header_bg, title_color, bar_fill = _PANEL, _NAVY, _TEAL
    header = (
        f'<TR><TD COLSPAN="3" BGCOLOR="{header_bg}" ALIGN="LEFT">'
        f'<FONT POINT-SIZE="10" COLOR="{title_color}"><B>{title}</B></FONT>'
        '</TD></TR>'
    )
    rows = _prob_rows(marginal, fill=bar_fill)
    return _table(header + rows, bg="white", cellpadding=4, selected=selected)


def _build_digraph(
    marginals: Mapping[str, Mapping[str, float]],
    *,
    observed: Mapping[str, str],
    observed_day: Mapping[str, int],
    edges: Iterable[tuple],
    selected: Optional[str],
    dpi: int,
) -> graphviz.Digraph:
    """Assemble the tabular BN ``Digraph`` shared by the PNG and clickable
    renderers, so the image and its click-map come from one identical graph."""
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
            root_style=_ROOT_DRIVER_COLORS.get(node),
            selected=(node == selected),
        )
        dot.node(node, label=label)

    for src, dst in edges:
        color = _TEAL if src in observed else "#94A3B8"
        dot.edge(src, dst, style="solid", color=color)

    return dot


def render_network_png(
    marginals: Mapping[str, Mapping[str, float]],
    *,
    observed: Mapping[str, str] = {},
    observed_day: Mapping[str, int] = {},
    edges: Optional[Iterable[tuple]] = None,
    selected: Optional[str] = None,
    dpi: int = 220,
) -> bytes:
    """Render the BN as PNG bytes for display in Streamlit.

    ``marginals`` must contain every node; ``observed`` maps evidence
    nodes to their set state; ``observed_day`` maps those same nodes
    to the day-of-session they were first set. ``edges`` selects the
    topology to draw (defaults to the labelling ``EDGES``); ``selected``
    draws an accent border on one node.
    """
    edges = list(EDGES if edges is None else edges)
    dot = _build_digraph(
        marginals, observed=observed, observed_day=observed_day,
        edges=edges, selected=selected, dpi=dpi,
    )
    return dot.pipe(format="png")


def _png_dimensions(png: bytes) -> Tuple[int, int]:
    """(width, height) in pixels from a PNG's IHDR chunk."""
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    width, height = struct.unpack(">II", png[16:24])
    return width, height


def _node_pixel_boxes(
    graph_json: str, png_w: int, png_h: int, dpi: int
) -> Dict[str, Tuple[float, float, float, float]]:
    """Map each node to its ``(x0, y0, x1, y1)`` box in PNG pixel space.

    Graphviz JSON reports node centres in points (origin bottom-left) and
    sizes in inches. The PNG adds a uniform pad around the layout ``bb``; we
    recover that pad from the actual PNG size so the mapping is exact
    regardless of graphviz's default margin, then flip y (PNG is top-down).
    """
    j = json.loads(graph_json)
    bb = [float(v) for v in j["bb"].split(",")]  # x0,y0,x1,y1 in points
    bb_w_pt, bb_h_pt = bb[2] - bb[0], bb[3] - bb[1]
    scale = dpi / 72.0
    # png_*_pt is the full canvas in points incl. pad; (png_pt - bb)/2 is the pad.
    pad_x = (png_w / scale - bb_w_pt) / 2.0
    pad_y = (png_h / scale - bb_h_pt) / 2.0
    png_h_pt = png_h / scale

    boxes: Dict[str, Tuple[float, float, float, float]] = {}
    for obj in j.get("objects", []):
        name, pos = obj.get("name"), obj.get("pos")
        if name not in STATES or pos is None or "width" not in obj:
            continue
        px, py = (float(v) for v in pos.split(","))
        w_px = float(obj["width"]) * dpi  # inches → px
        h_px = float(obj["height"]) * dpi
        cx = ((px - bb[0]) + pad_x) * scale
        cy = (png_h_pt - ((py - bb[1]) + pad_y)) * scale  # flip y
        boxes[name] = (cx - w_px / 2, cy - h_px / 2, cx + w_px / 2, cy + h_px / 2)
    return boxes


def render_network_clickable(
    marginals: Mapping[str, Mapping[str, float]],
    *,
    observed: Mapping[str, str] = {},
    observed_day: Mapping[str, int] = {},
    edges: Optional[Iterable[tuple]] = None,
    selected: Optional[str] = None,
    dpi: int = 200,
) -> Tuple[bytes, Dict[str, Tuple[float, float, float, float]], int, int]:
    """Render the BN once and return ``(png, node_boxes, width, height)``.

    ``node_boxes`` are pixel bounding boxes in the returned PNG's own
    coordinate space; a caller scales a click by ``width / displayed_width``
    and tests containment to resolve the clicked node. The PNG and the JSON
    geometry come from the *same* ``Digraph`` so the boxes line up exactly.
    """
    edges = list(EDGES if edges is None else edges)
    dot = _build_digraph(
        marginals, observed=observed, observed_day=observed_day,
        edges=edges, selected=selected, dpi=dpi,
    )
    png = dot.pipe(format="png")
    graph_json = dot.pipe(format="json").decode()
    png_w, png_h = _png_dimensions(png)
    boxes = _node_pixel_boxes(graph_json, png_w, png_h, dpi)
    return png, boxes, png_w, png_h


def node_at_pixel(
    boxes: Mapping[str, Tuple[float, float, float, float]], x: float, y: float
) -> Optional[str]:
    """Return the node whose box contains ``(x, y)`` in PNG pixels, else None."""
    for node, (x0, y0, x1, y1) in boxes.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return node
    return None


# ---------------------------------------------------------------------------
# Interactive (streamlit-agraph) payload
# ---------------------------------------------------------------------------

# Hierarchical "levels" for the vis.js hierarchical layout — roots on the
# left, scenario on the right. Chosen manually for a stable left-to-right
# DAG layout that does not jitter between reruns.
# (_ROOT_DRIVER_COLORS is defined near the palette above — shared by both
# renderers.)

# Layout (TOPOLOGY_LAYOUT) and the display-name overrides / title-wrap maps are
# scenario-specific and come from the active pack via the src.scenario seam.


def _display_name(raw: str) -> str:
    """Pack display override if any, else the generic underscore→space form."""
    return DISPLAY_OVERRIDES.get(raw) or raw.replace("_", " ")


def _wrap_node_title(name: str) -> str:
    """Two-line wrap for long node titles (pack-provided); identity otherwise."""
    return NODE_TITLE_WRAP.get(name, name)


def _format_label_text(
    node: str,
    marginal: Mapping[str, float],
    observed_state: Optional[str],
    day: Optional[int],
) -> str:
    """Build a monospace 'mini-table' label: title, rule, rows."""
    if node == LATENT:
        title_lines = ["SCENARIO"]
    else:
        title_lines = _wrap_node_title(_display_name(node)).split("\n")

    if observed_state is not None:
        day_suffix = f" · day {day}" if day is not None else ""
        body_lines = [f"● {_display_name(observed_state)}{day_suffix}"]
    else:
        states = list(STATES[node])
        display_states = [_display_name(s) for s in states]
        name_w = max(len(s) for s in display_states)
        body_lines = []
        for raw, disp in zip(states, display_states):
            pct = marginal.get(raw, 0.0) * 100
            body_lines.append(f"{disp.ljust(name_w)}  {pct:5.1f}%")

    width = max(len(s) for s in (*title_lines, *body_lines))
    separator = "─" * width
    return "\n".join([*title_lines, separator, *body_lines])


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
    edges: Optional[Iterable[tuple]] = None,
    node_level: Optional[Mapping[str, int]] = None,
    edge_titles: Optional[Mapping[tuple, str]] = None,
):
    """Return ``(nodes, edges, config)`` for ``streamlit_agraph.agraph``.

    The dashboard uses this for the interactive network view: nodes are
    clickable (returns the clicked node id), and vis.js provides pan /
    zoom / hover natively. ``edges`` / ``node_level`` select the topology
    (default: the labelling ``EDGES`` / ``_NODE_LEVEL``).
    """
    # Lazy-import so the rest of the package keeps working without the dep.
    from streamlit_agraph import Config, Edge, Node

    edge_list = list(EDGES if edges is None else edges)
    levels = TOPOLOGY_LAYOUT["labelling"][1] if node_level is None else node_level

    nodes = []
    for node in STATES.keys():
        marginal = marginals[node]
        obs_state = observed.get(node)
        day = observed_day.get(node)
        is_root_driver = node in _ROOT_DRIVER_COLORS

        if node == LATENT:
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
                    "size": 17,
                    "face": "Menlo, Consolas, 'Courier New', monospace",
                    "multi": False,
                    "align": "left",
                },
                borderWidth=3 if is_root_driver else (2 if obs_state is not None else 1),
                level=levels.get(node, 0),
                margin=4,
            )
        )

    titles = edge_titles or {}
    edges = []
    for src, dst in edge_list:
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
                title=titles.get((src, dst), ""),  # vis.js hover tooltip (P10 / C11)
            )
        )

    # Canvas height calibrated to the combined height of the Posterior
    # and Override boxes in the right column of the Network tab.
    # nodeSpacing / treeSpacing scaled proportionally from the original
    # 240/260 @ height=340 so the DAG fills the canvas.
    config = Config(
        width="100%",
        height=460,
        directed=True,
        physics=False,
        hierarchical=True,
        nodeHighlightBehavior=True,
        highlightColor=_TEAL,
        collapsible=False,
        node={"labelProperty": "label", "renderLabel": True},
        link={"renderLabel": False},
        layout={
            "hierarchical": {
                "enabled": True,
                "direction": "LR",
                "sortMethod": "directed",
                "levelSeparation": 330,
                "nodeSpacing": 300,
                "treeSpacing": 325,
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


__all__ = [
    "render_network_png",
    "render_network_clickable",
    "node_at_pixel",
    "build_agraph_payload",
    "TOPOLOGY_LAYOUT",
]
