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
    soft_assignments: Dict[str, Dict[str, float]] = field(default_factory=dict)
    rationale: str = ""                    # translator's overall_rationale, if any
    per_assignment_reasons: Dict[str, str] = field(default_factory=dict)
    source: Literal["translator", "manual"] = "translator"


@dataclass(frozen=True)
class ExampleHeadline:
    """A plain-text headline the user can inject with one click."""

    text: str
    tone: Tone


# Per-scenario example headlines now live in the pack (e.g.
# packs/hormuz/headlines.py) and are reached via the ``src.scenario`` seam.

__all__ = ["Observation", "ExampleHeadline", "Tone"]
