"""Elicitation platform (Plan 4).

A Cooke-protocol CPT elicitation platform with AI-only expert panels,
per-CPT calibration-derived kappa, and full provenance. See
``docs/04_elicitation_tool_plan.md`` for the executable plan and
``docs/elicitation_methodology_and_defensibility.md`` for the methodology.

This package is built foundation-first. Layer 0 (this commit) is the data
model and storage substrate: per-deployment configuration, LLM provider
credentials (deployment key / OAuth / policy-gated BYOK), and the database
schema.
"""

from __future__ import annotations

__all__: list[str] = []
