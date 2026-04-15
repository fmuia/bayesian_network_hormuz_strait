"""Static matplotlib renderer for the BN structure."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx

from .network import EDGES, STATES

# Hand-tuned hierarchical layout. Coordinates are (x, y) in arbitrary
# units; matplotlib + networkx will scale them. Roots on the left,
# scenario terminal on the right.
_LAYOUT = {
    "US_Iran_Negotiations":         (0.0, 3.5),
    "Iranian_Regime_Stability":     (0.0, 2.0),
    "Third_Party_Mediation":        (0.0, 0.5),
    "Sanctions_Trajectory":         (0.0, -1.0),
    "Iranian_Proxy_Activity":       (1.5, 1.0),
    "Tanker_Incidents":             (3.0, 2.0),
    "US_Military_Response":         (4.5, 0.5),
    "Strait_Operationally_Closed":  (6.0, 1.5),
    "Energy_Infrastructure_Damage": (7.5, 2.5),
    "Conflict_Duration":            (6.0, -0.5),
    "Diplomatic_Resolution_Path":   (4.5, -2.0),
    "Oil_Price_Regime":             (9.0, 0.0),
    "Scenario":                     (10.5, 1.0),
}

# Restrained palette aligned with the dashboard.
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

    fig, ax = plt.subplots(figsize=(11, 6.5))
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

    nx.draw_networkx_edges(
        g, _LAYOUT, ax=ax,
        edge_color="#94A3B8", arrows=True, arrowsize=12,
        width=1.0, node_size=2400,
    )
    nx.draw_networkx_nodes(
        g, _LAYOUT, ax=ax,
        node_color=node_colors, edgecolors=edgecolors,
        linewidths=1.5, node_size=2400,
    )
    # Wrap long labels for readability.
    labels = {n: n.replace("_", "\n", 1).replace("_", " ") for n in g.nodes()}
    for node, (x, y) in _LAYOUT.items():
        ax.text(
            x, y, labels[node],
            ha="center", va="center",
            fontsize=7.5, color=text_colors[list(g.nodes()).index(node)],
            fontweight="600",
        )

    ax.set_xlim(-1.0, 11.5)
    ax.set_ylim(-3.0, 4.5)
    ax.axis("off")
    fig.tight_layout()
    return fig


__all__ = ["render_network"]
