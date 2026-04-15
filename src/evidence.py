"""Observation record and a handful of example headlines for the demo.

The old hardcoded news catalogue has been retired: headlines now flow
through the LLM translator (see `src.translator`). We keep a short
list of example strings purely as one-click seeds to drive the demo in
a meeting — they are plain text, with *no* pre-baked node mappings,
so they exercise the same translation path a live headline would.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

Tone = Literal["de-escalation", "mixed", "escalation"]


@dataclass
class Observation:
    """One committed observation in the session log."""

    day: int
    headline: str
    assignments: Dict[str, str]            # {node: state}
    rationale: str = ""                    # translator's overall_rationale, if any
    per_assignment_reasons: Dict[str, str] = field(default_factory=dict)
    source: Literal["translator", "manual"] = "translator"


@dataclass(frozen=True)
class ExampleHeadline:
    """A plain-text headline the user can inject with one click."""

    text: str
    tone: Tone


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


__all__ = ["Observation", "ExampleHeadline", "EXAMPLE_HEADLINES"]
