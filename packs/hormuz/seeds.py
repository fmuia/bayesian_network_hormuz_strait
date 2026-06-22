"""Hormuz elicitation calibration seeds (moved from
src/elicitation/integration/seeds.py).

(id, text, realization, unit). The realization is used only for Cooke scoring.
These probe the same *kind* of judgment as the CPT targets — incident counts,
disruption durations, premium/price spikes — per methodology §8.3. Illustrative;
replace realizations with a vetted set before high-stakes use.
"""
from __future__ import annotations

ELICITATION_SEEDS = [
    ("hormuz_closure_days", "Estimate the number of days the Strait of Hormuz has been fully closed to maritime traffic in the past 40 years", 0.0, "days"),
    ("closure_threats_15y", "Estimate the number of distinct occasions on which Iranian officials have publicly threatened to close the Strait of Hormuz over the past ~15 years", 8.0, "count"),
    ("vessels_attacked_2019", "Estimate the number of commercial vessels attacked or seized in or near the Strait of Hormuz and Gulf of Oman during 2019", 6.0, "count"),
    ("war_risk_premium_2019", "Estimate the Gulf war-risk insurance premium for a tanker transit at the mid-2019 peak, as a percent of the ship's hull value", 0.4, "% of hull value"),
    ("brent_jump_abqaiq", "Estimate the single-trading-day percent rise in Brent crude immediately after the September 2019 Abqaiq facility attack", 15.0, "%"),
    ("abqaiq_supply_removed", "Estimate the peak crude-oil supply removed from the market by the September 2019 Abqaiq/Khurais attack", 5.7, "million bbl/day"),
]

__all__ = ["ELICITATION_SEEDS"]
