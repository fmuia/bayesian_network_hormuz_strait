"""T03 — golden-set eval harness tests (offline, no live LLM).

The contract gate: every golden record's recorded response still validates to a
well-formed A1/A2 result. Accuracy metrics are computed but only sanity-checked
(not thresholded) at this corpus size.
"""
from __future__ import annotations

from scripts.translator_eval import evaluate, load_golden, _predicted


def test_golden_records_present():
    recs = load_golden()
    assert len(recs) >= 8  # seed size (grows to 30 -> 50 before T08)


def test_every_recorded_response_satisfies_contract():
    """Gated regression: each recorded response is still a valid A1/A2 result."""
    for r in load_golden():
        pred = _predicted(r)  # raises TranslatorError on any contract violation
        assert isinstance(pred, dict)
        # every predicted state is a legal state for its node (A2)
        from src.scenario import STATES
        for node, state in pred.items():
            assert state in STATES[node]


def test_eval_metrics_computable_and_bounded():
    m = evaluate(load_golden())
    assert m["n_records"] == len(load_golden())
    for key in ("node_precision", "node_recall", "node_f1",
                "abstention_precision", "abstention_recall"):
        val = m[key]
        assert val is None or 0.0 <= val <= 1.0


def test_node_coverage_is_broad():
    """Design intent: every observable node covered >=1x. Seed covers all 12."""
    m = evaluate(load_golden())
    assert m["n_nodes_covered"] >= m["n_observable_nodes"]  # full coverage at seed


def test_offtopic_record_abstains():
    recs = {r["id"]: r for r in load_golden()}
    assert "offtopic_football" in recs
    assert _predicted(recs["offtopic_football"]) == {}  # empty -> abstain
