"""Calibration seed loading/saving for the Cooke protocol.

``default_seeds()`` resolves the *active pack's* seed set (``packs/<id>/seeds.py``)
via the src.scenario seam — each scenario ships domain-matched calibration
questions (methodology §8.3: calibration only transfers if the seeds probe the
same judgment as the CPT targets). The seeds are *illustrative*, retrodictive
bootstraps (a provisional day-one weight, not a certificate); realised values are
commonly-cited approximations and must be replaced with a vetted set before
high-stakes use. They remain editable in the UI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..protocols.base import SeedQuestion

def default_seeds() -> list[SeedQuestion]:
    """Calibration seeds for the active scenario pack (Cooke protocol). Each pack
    ships its own domain-matched seed set (``packs/<id>/seeds.py``), resolved via
    the src.scenario seam so it follows ``SCENARIO_PACK``."""
    from src.scenario import ELICITATION_SEEDS

    return [SeedQuestion(*row) for row in ELICITATION_SEEDS]


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


__all__ = ["default_seeds", "slug_id", "save_seeds", "load_seeds"]
