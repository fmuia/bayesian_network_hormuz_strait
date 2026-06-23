"""Meridian one-click example headlines (rare-earth-magnet escalation arc)."""
from __future__ import annotations

from typing import List

from src.evidence import ExampleHeadline

EXAMPLE_HEADLINES: List[ExampleHeadline] = [
    ExampleHeadline(
        "Suppliers report stable lead times; spot magnet prices flat week-on-week",
        "de-escalation",
    ),
    ExampleHeadline(
        "China expands rare-earth export licensing to cover dysprosium and terbium",
        "escalation",
    ),
    ExampleHeadline(
        "Magnet PO lead times from key supplier stretch from 6 to 14 weeks",
        "escalation",
    ),
    ExampleHeadline(
        "Spot NdFeB magnet prices jump 35% as buyers scramble for inventory",
        "escalation",
    ),
    ExampleHeadline(
        "Three tier-2 suppliers declare force majeure after regional port closure",
        "escalation",
    ),
    ExampleHeadline(
        "Trade talks ease export curbs; licences issued, lead times begin to normalise",
        "de-escalation",
    ),
]

__all__ = ["EXAMPLE_HEADLINES"]
