"""Static matplotlib renderer for the BN structure."""

from __future__ import annotations

from typing import Iterable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from .network import EDGES, STATES

# Hand-tuned hierarchical layout. Coordinates are (x, y) in arbitrary
# units; matplotlib + networkx will scale them. Roots on the left,
# scenario terminal on the right. Spacing has been widened so labels
# don't overlap when rendered at projector size.
_LAYOUT = {
    "US_Iran_Negotiations":         (0.0,  4.5),
    "Iranian_Regime_Stability":     (0.0,  2.5),
    "Third_Party_Mediation":        (0.0,  0.5),
    "Sanctions_Trajectory":         (0.0, -1.5),
    "Iranian_Proxy_Activity":       (3.0,  1.5),
    "Tanker_Incidents":             (5.5,  3.2),
    "US_Military_Response":         (8.0,  0.8),
    "Strait_Operationally_Closed":  (10.5, 2.5),
    "Energy_Infrastructure_Damage": (13.0, 3.6),
    "Conflict_Duration":            (10.5, -1.0),
    "Diplomatic_Resolution_Path":   (8.0,  -3.2),
    "Oil_Price_Regime":             (15.5, 0.8),
    "Scenario":                     (17.8, 1.5),
}

_TEAL = "#1A7A6D"
_NAVY = "#1B2A3D"
_LIGHT = "#F5F5F5"
_SCENARIO_FILL = "#1B2A3D"


def render_network(observed_nodes: Iterable[str] = ()) -> plt.Figure:
    """Render the BN with `observed_nodes` highlighted in teal."""
    observed = set(observed_nodes)
    g = nx.DiGraph()
    g.add_nodes_from(STATES.keys())
    g.add_edges_from(EDGES)

    fig, ax = plt.subplots(figsize=(18, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    node_colors = []
    edgecolors = []
    text_colors = []
    for node in g.nodes():
        if node == "Scenario":
            node_colors.append(_SCENARIO_FILL)
            edgecolors.append(_SCENARIO_FILL)
            text_colors.append("white")
        elif node in observed:
            node_colors.append(_TEAL)
            edgecolors.append(_TEAL)
            text_colors.append("white")
        else:
            node_colors.append(_LIGHT)
            edgecolors.append(_NAVY)
            text_colors.append(_NAVY)

    node_size = 7200
    nx.draw_networkx_edges(
        g, _LAYOUT, ax=ax,
        edge_color="#94A3B8", arrows=True, arrowsize=20,
        width=1.4, node_size=node_size,
        connectionstyle="arc3,rad=0.04",
    )
    nx.draw_networkx_nodes(
        g, _LAYOUT, ax=ax,
        node_color=node_colors, edgecolors=edgecolors,
        linewidths=2.0, node_size=node_size,
    )
    node_list = list(g.nodes())
    for idx, node in enumerate(node_list):
        x, y = _LAYOUT[node]
        # Split long names at the first underscore to get two tidy lines.
        label = node.replace("_", " ")
        if len(label) > 14:
            words = label.split(" ")
            mid = len(words) // 2
            label = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        ax.text(
            x, y, label,
            ha="center", va="center",
            fontsize=11, color=text_colors[idx],
            fontweight="600",
        )

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=_LIGHT, edgecolor=_NAVY, label="Unobserved"),
        mpatches.Patch(facecolor=_TEAL, edgecolor=_TEAL, label="Observed"),
        mpatches.Patch(facecolor=_SCENARIO_FILL, edgecolor=_SCENARIO_FILL,
                       label="Terminal scenario"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower left",
        frameon=False, fontsize=10,
    )

    ax.set_xlim(-1.8, 19.5)
    ax.set_ylim(-4.5, 5.8)
    ax.axis("off")
    fig.tight_layout()
    return fig


__all__ = ["render_network"]
