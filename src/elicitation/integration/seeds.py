"""Illustrative Hormuz calibration seed set.

TODO(pack-separation): this module is Hormuz-specific content sitting in shared
code — relocate to packs/hormuz/ and load per active pack (elicitation Phase D).


These are *illustrative* calibration questions used to score the panel (Cooke).
They are chosen to match the **judgment the targets require**: the CPTs are
probabilistic judgments about *crisis dynamics* (incidents, escalation, market
and supply consequences), so the seeds ask about the same kind of thing —
incident counts, disruption durations, threat frequencies, premium and price
spikes — rather than static energy-system magnitudes (VLCC size, flow volumes),
which test numeracy the targets do not need. This is the relevance constraint of
methodology §8.3: calibration only transfers if the seeds probe the same
judgment as the targets.

They are framed as *estimation* tasks and the prompt asks the agent to flag any
answer it is merely *recalling*; such answers are discarded at scoring time
(source-attribution, §8.3). But the questions are still **retrodictive** (past,
resolved events an LLM may have memorised), so they remain a *bootstrap* — a
provisional day-one weight, not a certificate. The contamination-proof upgrade is
**prospective** seeds (crisis questions that resolve in the future); see the
plan's open items. Realised values are commonly-cited approximations and must be
replaced with a vetted set before high-stakes use. They are editable in the UI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..protocols.base import SeedQuestion

# TODO(pack-separation): this entire DEFAULT_SEEDS set is Hormuz-specific calibration
# content — move it into packs/hormuz/ and make default_seeds() resolve per active pack
# (elicitation Phase D). Each pack ships its own domain-matched seed questions.
DEFAULT_SEEDS: list[SeedQuestion] = [
    SeedQuestion("hormuz_closure_days", "Estimate the number of days the Strait of Hormuz has been fully closed to maritime traffic in the past 40 years", 0.0, "days"),
    SeedQuestion("closure_threats_15y", "Estimate the number of distinct occasions on which Iranian officials have publicly threatened to close the Strait of Hormuz over the past ~15 years", 8.0, "count"),
    SeedQuestion("vessels_attacked_2019", "Estimate the number of commercial vessels attacked or seized in or near the Strait of Hormuz and Gulf of Oman during 2019", 6.0, "count"),
    SeedQuestion("war_risk_premium_2019", "Estimate the Gulf war-risk insurance premium for a tanker transit at the mid-2019 peak, as a percent of the ship's hull value", 0.4, "% of hull value"),
    SeedQuestion("brent_jump_abqaiq", "Estimate the single-trading-day percent rise in Brent crude immediately after the September 2019 Abqaiq facility attack", 15.0, "%"),
    SeedQuestion("abqaiq_supply_removed", "Estimate the peak crude-oil supply removed from the market by the September 2019 Abqaiq/Khurais attack", 5.7, "million bbl/day"),
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
