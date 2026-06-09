"""Illustrative Hormuz calibration seed set.

These are *illustrative* known-answer questions used to score the panel's
calibration (Cooke). They are relevance-constrained (Strait-of-Hormuz / Gulf
domain) but their realised values are commonly-cited approximations and must be
replaced with a vetted, analyst-authored set before any high-stakes use. They
are editable in the elicitation framework.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..protocols.base import SeedQuestion

DEFAULT_SEEDS: list[SeedQuestion] = [
    SeedQuestion("hormuz_oil_share", "Percent of global seaborne crude oil that transits the Strait of Hormuz in recent years", 21.0, "%"),
    SeedQuestion("hormuz_flow_mbd", "Approximate oil flow through the Strait of Hormuz in recent years", 21.0, "million bbl/day"),
    SeedQuestion("hormuz_width_km", "Narrowest width of the Strait of Hormuz", 33.0, "km"),
    SeedQuestion("tanker_war_year", "Calendar year the Iran-Iraq 'Tanker War' attacks on shipping began", 1984.0, "year"),
    SeedQuestion("gulf_states_count", "Number of countries with a Persian Gulf coastline", 8.0, "count"),
    SeedQuestion("gulf_oman_attacks_year", "Year of the widely-reported Gulf of Oman tanker attacks that spiked war-risk premiums", 2019.0, "year"),
]


def default_seeds() -> list[SeedQuestion]:
    return list(DEFAULT_SEEDS)


def slug_id(text: str, index: int) -> str:
    """A readable, unique-per-row id for an analyst-authored seed."""
    base = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "seed"
    return f"{base}_{index}"


def save_seeds(seeds: list[SeedQuestion], path: str | Path) -> Path:
    """Persist a seed set as JSON (the deployment's calibration questions)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [{"id": s.id, "text": s.text, "realization": s.realization, "unit": s.unit} for s in seeds],
            indent=2,
        )
    )
    return path


def load_seeds(path: str | Path) -> list[SeedQuestion]:
    """Load a saved seed set, or [] if none has been authored yet."""
    path = Path(path)
    if not path.is_file():
        return []
    return [SeedQuestion(**d) for d in json.loads(path.read_text())]


__all__ = ["DEFAULT_SEEDS", "default_seeds", "slug_id", "save_seeds", "load_seeds"]
