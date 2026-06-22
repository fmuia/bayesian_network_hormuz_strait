"""Guard: Hormuz-specific STRUCTURE must not leak into shared engine/UI code.

The shared tree (``src/`` + ``app/``, excluding the seam-fed packs) must contain
no Hormuz node names or scenario-state names — those are exactly the literals
that would break a different active pack (e.g. Meridian). They belong only in
``packs/hormuz/``. This test makes the separation self-enforcing: add a stray
Hormuz identifier to a shared module and CI goes red.

Scope note: this guards STRUCTURAL identifiers (node + latent-state names). Domain
*flavour text* (e.g. an elicitation seed mentioning "Hormuz") is tracked
separately and intentionally not covered here.
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHARED_DIRS = [ROOT / "src", ROOT / "app"]

# The 13 Hormuz node names + 3 latent (Scenario) state names.
HORMUZ_STRUCTURAL_TOKENS = [
    "US_Iran_Negotiations", "Iranian_Regime_Stability", "Third_Party_Mediation",
    "Sanctions_Trajectory", "Iran_Aligned_Militia_Attacks", "Tanker_Incidents",
    "US_Military_Response", "Strait_Operationally_Closed",
    "Energy_Infrastructure_Damage", "Conflict_Duration",
    "Diplomatic_Resolution_Path", "Oil_Price_Regime",
    "Stress_Mitigates", "Prolonged_Conflict", "Severe_Closure",
]


def _shared_py_files():
    for d in SHARED_DIRS:
        for path in d.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


def test_no_hormuz_structure_in_shared_code():
    offenders = {}
    for path in _shared_py_files():
        text = path.read_text(encoding="utf-8")
        hits = sorted({t for t in HORMUZ_STRUCTURAL_TOKENS if t in text})
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits
    assert not offenders, (
        "Hormuz structural identifiers leaked into shared code (move them to "
        f"packs/hormuz/ and read via the src.scenario seam): {offenders}"
    )
