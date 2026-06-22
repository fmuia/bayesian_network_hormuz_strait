"""Hormuz one-click example headlines (moved out of src/evidence.py).

Plain-text seeds with no pre-baked node mappings — they exercise the same
translation path a live headline would. The ``ExampleHeadline`` container is the
generic type from ``src.evidence``; only this list is Hormuz-specific.
"""
from __future__ import annotations

from typing import List

from src.evidence import ExampleHeadline

EXAMPLE_HEADLINES: List[ExampleHeadline] = [
    ExampleHeadline(
        "Oman confirms active US–Iran back-channel talks in Muscat",
        "de-escalation",
    ),
    ExampleHeadline(
        "Treasury issues 90-day sanctions waiver for Iranian oil exports",
        "de-escalation",
    ),
    ExampleHeadline(
        "IRGC announces 'inspection regime' on all Hormuz traffic",
        "mixed",
    ),
    ExampleHeadline(
        "Fourth tanker incident in two weeks; insurers raise war-risk premia",
        "escalation",
    ),
    ExampleHeadline(
        "US conducts strikes against IRGC naval assets after tanker attack",
        "escalation",
    ),
    ExampleHeadline(
        "Major fire at Ras Tanura terminal after missile strike; strait closed",
        "escalation",
    ),
]

__all__ = ["EXAMPLE_HEADLINES"]
