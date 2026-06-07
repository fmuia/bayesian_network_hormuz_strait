"""Translator golden-set evaluation harness.

Placeholder (Plan 2 / T00). The real harness — node recall/precision, state
accuracy, Brier on the likelihood vectors, abstention precision/recall, plus the
metrics snapshot the dashboard badge reads — is wired up in T03 against the
golden set in ``tests/golden/translator/``.

Run with ``pixi run translator-eval``.
"""
from __future__ import annotations

import sys


def main() -> int:
    print(
        "translator-eval: placeholder (T00). The evaluation harness is wired up "
        "in T03 against tests/golden/translator/. No metrics to report yet."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
