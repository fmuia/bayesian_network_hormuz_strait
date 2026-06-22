"""Shared palette + scenario labels for the dashboard and its components.

Extracted from ``app/dashboard.py`` (Plan 5 P3) so the chart components can use
the palette without importing the dashboard (which would be circular). The CSS
counterpart is ``app/styles.css`` (P1) — keep the hex values in sync. Plan 5 C9
swaps these for the Wong CVD-safe set.
"""
from __future__ import annotations

from src.scenario import PACK as _PACK

TEAL = "#1A7A6D"
NAVY = "#1B2A3D"
PANEL = "#F5F5F5"
RULE = "#E5E7EB"
MUTED = "#6B7280"
GREEN = "#2E8B57"
AMBER = "#D4A017"
RED = "#B22222"

# Scenario (latent-state) display data + root-driver styling come from the active
# pack's presentation layer; the palette above stays shared. SCENARIO_KEYS is the
# latent states in their declared order.
SCENARIO_KEYS = tuple(_PACK.states[_PACK.latent])
SCENARIO_COLOR = _PACK.presentation.scenario_color
SCENARIO_LABEL = _PACK.presentation.scenario_label
ROOT_DRIVER_STYLE = _PACK.presentation.root_driver_style
