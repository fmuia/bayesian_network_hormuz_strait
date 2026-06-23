"""Structured CPT builders — generate a full CPT from a few elicited inputs.

Pack-agnostic machinery. ``ranked_node_cpt`` implements the ranked-node /
weighted-severity method (Fenton et al.): instead of eliciting every parent
configuration, the modeller supplies (a) a per-parent weight and (b) a handful of
anchor distributions along a 0..1 severity axis; the full table is interpolated.
This is exactly the structured-elicitation technique that makes a large latent
CPT (e.g. 3 ordered drivers → 27 columns) tractable to elicit.
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Mapping, Sequence, Tuple


def _interp_anchor(z: float, anchors: Sequence[Tuple[float, Sequence[float]]]) -> List[float]:
    """Linear-interpolate the child distribution at severity ``z`` from anchors
    (sorted by their z). Clamps below/above the anchor range. A convex blend of
    two distributions is itself a distribution; renormalised defensively."""
    pts = sorted(anchors, key=lambda a: a[0])
    if z <= pts[0][0]:
        out = list(pts[0][1])
    elif z >= pts[-1][0]:
        out = list(pts[-1][1])
    else:
        out = list(pts[-1][1])
        for (z0, d0), (z1, d1) in zip(pts, pts[1:]):
            if z0 <= z <= z1:
                t = (z - z0) / (z1 - z0) if z1 > z0 else 0.0
                out = [a + t * (b - a) for a, b in zip(d0, d1)]
                break
    s = sum(out)
    return [x / s for x in out]


def ranked_node_cpt(
    parents: Sequence[str],
    parent_states: Mapping[str, Sequence[str]],
    weights: Mapping[str, float],
    child_states: Sequence[str],
    anchors: Sequence[Tuple[float, Sequence[float]]],
) -> Dict[Tuple[str, ...], List[float]]:
    """Full ``{parent-state-tuple: child-distribution}`` CPT via the ranked-node
    method.

    Each parent state maps to a normalised rank in ``[0, 1]`` (first state 0,
    last state 1); the weighted sum of ranks (``weights`` should sum to ~1) gives
    a severity ``z`` whose child distribution is interpolated from ``anchors``
    (each ``(z, [p over child_states])``; provide at least the z=0 and z=1 ends).
    """
    if abs(sum(weights[p] for p in parents) - 1.0) > 1e-6:
        raise ValueError("ranked-node parent weights must sum to 1.0")
    for _z, dist in anchors:
        if len(dist) != len(child_states):
            raise ValueError("each anchor distribution must cover all child states")
    cpt: Dict[Tuple[str, ...], List[float]] = {}
    for combo in product(*[list(parent_states[p]) for p in parents]):
        z = 0.0
        for parent, state in zip(parents, combo):
            states = list(parent_states[parent])
            rank = states.index(state) / (len(states) - 1) if len(states) > 1 else 0.0
            z += weights[parent] * rank
        cpt[tuple(combo)] = _interp_anchor(z, anchors)
    return cpt


__all__ = ["ranked_node_cpt"]
