"""Translator golden-set evaluation harness (Plan 2 / T03 / D2).

Runs **offline**: each golden record carries a ``recorded_response`` (a real
model payload captured once), which is replayed through the *same*
``_validate_payload`` path the live translator uses. No network, deterministic.

At this corpus size the harness **gates contract regressions only** — every
recorded response must still validate to a well-formed A1/A2 result. The
accuracy/calibration numbers are reported with an explicit ``n`` and are *not*
gated (they are statistically meaningless on a ~10-record seed; calibration /
Brier are deferred until the corpus reaches ~100, see R-cal in the commit plan).

Run with ``pixi run translator-eval`` (prints metrics + writes the snapshot the
dashboard badge reads).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path

from src.scenario import STATES  # noqa: E402
from src.translator import TranslatorError, _validate_payload  # noqa: E402

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "tests" / "golden" / "translator"
SNAPSHOT_PATH = GOLDEN_DIR / "_eval_snapshot.json"
_N_OBSERVABLE = len([n for n in STATES if n != "Scenario"])


def load_golden() -> List[Dict]:
    """Load every golden record (``*.json`` except ``_``-prefixed meta files)."""
    records: List[Dict] = []
    for path in sorted(GOLDEN_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data["_file"] = path.name
        records.append(data)
    return records


def _predicted(record: Dict) -> Dict[str, str]:
    """Replay the recorded response through the validator -> {node: top state}.

    Raises ``TranslatorError`` if the recorded response no longer satisfies the
    A1/A2 contract — that is the gated regression.
    """
    assignments, _ = _validate_payload(record["recorded_response"])
    return {a.node: a.state for a in assignments}


def _expected(record: Dict) -> Dict[str, str]:
    return {a["node"]: a["state"] for a in record["expected"]["assignments"]}


def _ratio(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def evaluate(records: List[Dict]) -> Dict:
    nodes_tp = nodes_fp = nodes_fn = 0
    state_match = state_total = 0
    abst_tp = abst_fp = abst_fn = 0
    covered: set = set()
    per_record: List[Dict] = []

    for r in records:
        pred = _predicted(r)          # raises on contract violation
        exp = _expected(r)
        covered |= set(exp)
        pred_nodes, exp_nodes = set(pred), set(exp)
        nodes_tp += len(pred_nodes & exp_nodes)
        nodes_fp += len(pred_nodes - exp_nodes)
        nodes_fn += len(exp_nodes - pred_nodes)
        for node in pred_nodes & exp_nodes:
            state_total += 1
            state_match += int(pred[node] == exp[node])
        pred_abst = not pred_nodes
        exp_abst = r["expected"].get("relevance") == "no" or not exp_nodes
        abst_tp += int(exp_abst and pred_abst)
        abst_fp += int(pred_abst and not exp_abst)
        abst_fn += int(exp_abst and not pred_abst)
        per_record.append({"id": r["id"], "predicted": pred, "expected": exp})

    return {
        "n_records": len(records),
        "gate": "contract-only",
        "node_precision": _ratio(nodes_tp, nodes_tp + nodes_fp),
        "node_recall": _ratio(nodes_tp, nodes_tp + nodes_fn),
        "node_f1": _ratio(2 * nodes_tp, 2 * nodes_tp + nodes_fp + nodes_fn),
        "state_accuracy_given_node_match": _ratio(state_match, state_total),
        "abstention_precision": _ratio(abst_tp, abst_tp + abst_fp),
        "abstention_recall": _ratio(abst_tp, abst_tp + abst_fn),
        "nodes_covered": sorted(covered),
        "n_nodes_covered": len(covered),
        "n_observable_nodes": _N_OBSERVABLE,
        "per_record": per_record,
        "notes": (
            "Contract-only gate at this corpus size. Accuracy numbers are "
            "n-annotated, not gated. Brier/calibration deferred until corpus "
            ">= ~100 (R-cal)."
        ),
    }


def main() -> int:
    records = load_golden()
    if not records:
        print("translator-eval: no golden records found in", GOLDEN_DIR)
        return 0
    try:
        m = evaluate(records)
    except TranslatorError as exc:
        print("translator-eval: CONTRACT FAILURE —", exc)
        return 1
    SNAPSHOT_PATH.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    print(f"translator-eval: n={m['n_records']} (gate: {m['gate']})")
    print(f"  node P/R/F1            : {m['node_precision']} / {m['node_recall']} / {m['node_f1']}")
    print(f"  state acc | node-match : {m['state_accuracy_given_node_match']}")
    print(f"  abstention P/R         : {m['abstention_precision']} / {m['abstention_recall']}")
    print(f"  nodes covered          : {m['n_nodes_covered']}/{m['n_observable_nodes']}")
    print("  snapshot ->", SNAPSHOT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
