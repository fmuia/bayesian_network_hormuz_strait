"""Ranked-node CPT builder (src.cpt_builders)."""
from __future__ import annotations

import pytest

from src.cpt_builders import ranked_node_cpt

_PS = {"A": ["lo", "hi"], "B": ["lo", "hi"]}
_ANCHORS = [(0.0, [0.9, 0.1]), (1.0, [0.1, 0.9])]


def test_every_column_is_a_distribution():
    cpt = ranked_node_cpt(["A", "B"], _PS, {"A": 0.5, "B": 0.5}, ["good", "bad"], _ANCHORS)
    assert set(cpt) == {("lo", "lo"), ("lo", "hi"), ("hi", "lo"), ("hi", "hi")}
    for col in cpt.values():
        assert abs(sum(col) - 1.0) < 1e-9


def test_endpoints_hit_the_anchors():
    cpt = ranked_node_cpt(["A", "B"], _PS, {"A": 0.5, "B": 0.5}, ["good", "bad"], _ANCHORS)
    assert cpt[("lo", "lo")] == [0.9, 0.1]   # z=0
    assert cpt[("hi", "hi")] == [0.1, 0.9]   # z=1
    assert cpt[("lo", "hi")] == pytest.approx([0.5, 0.5])  # z=0.5 midpoint


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        ranked_node_cpt(["A", "B"], _PS, {"A": 0.3, "B": 0.3}, ["good", "bad"], _ANCHORS)


def test_anchor_arity_checked():
    with pytest.raises(ValueError, match="all child states"):
        ranked_node_cpt(["A"], {"A": ["lo", "hi"]}, {"A": 1.0}, ["x", "y", "z"], [(0.0, [1.0, 0.0])])
