"""Meridian elicitation calibration seeds.

(id, text, realization, unit). Domain-matched to the CPT targets — supply-chain
disruption magnitudes: lead-time blowouts, price spikes, force-majeure counts,
freight-cost surges (methodology §8.3). Illustrative realizations; replace with a
vetted set before high-stakes use.
"""
from __future__ import annotations

ELICITATION_SEEDS = [
    ("rare_earth_price_2011", "Estimate the peak multiple by which key rare-earth oxide prices rose during the 2010-2011 China export-quota shock (e.g. 5x, 10x)", 10.0, "x"),
    ("china_rare_earth_share", "Estimate China's approximate share of global rare-earth *refining/processing* capacity as of 2024", 90.0, "%"),
    ("suez_blockage_days", "Estimate the number of days the Suez Canal was blocked by the Ever Given grounding in March 2021", 6.0, "days"),
    ("auto_chip_units_lost_2021", "Estimate the number of vehicles global automakers could not build in 2021 due to the semiconductor shortage", 10.0, "million units"),
    ("scfi_peak_multiple_covid", "Estimate the peak multiple of the Shanghai Containerized Freight Index during 2021-2022 versus its pre-COVID 2019 baseline", 5.0, "x"),
    ("air_freight_premium", "Estimate the typical cost multiple of air freight versus ocean freight per kg for expedited recovery shipments", 12.0, "x"),
]

__all__ = ["ELICITATION_SEEDS"]
