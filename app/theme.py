"""Shared palette + scenario labels for the dashboard and its components.

Extracted from ``app/dashboard.py`` (Plan 5 P3) so the chart components can use
the palette without importing the dashboard (which would be circular). The CSS
counterpart is ``app/styles.css`` (P1) — keep the hex values in sync. Plan 5 C9
swaps these for the Wong CVD-safe set.
"""
from __future__ import annotations

TEAL = "#1A7A6D"
NAVY = "#1B2A3D"
PANEL = "#F5F5F5"
RULE = "#E5E7EB"
MUTED = "#6B7280"
GREEN = "#2E8B57"
AMBER = "#D4A017"
RED = "#B22222"

# Scenario states in display order (used by the cards and the evolution chart).
SCENARIO_KEYS = ("Stress_Mitigates", "Prolonged_Conflict", "Severe_Closure")

SCENARIO_COLOR = {
    "Stress_Mitigates": GREEN,
    "Prolonged_Conflict": AMBER,
    "Severe_Closure": RED,
}
SCENARIO_LABEL = {
    "Stress_Mitigates": "Stress Mitigates",
    "Prolonged_Conflict": "Prolonged Conflict",
    "Severe_Closure": "Severe Closure",
}
ROOT_DRIVER_STYLE = {
    "US_Iran_Negotiations": ("#DBEAFE", "#1D4ED8"),
    "Iranian_Regime_Stability": ("#FCE7F3", "#BE185D"),
    "Third_Party_Mediation": ("#FEF3C7", "#B45309"),
    "Sanctions_Trajectory": ("#EDE9FE", "#6D28D9"),
}
