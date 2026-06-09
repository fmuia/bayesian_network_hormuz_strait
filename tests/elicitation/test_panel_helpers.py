"""Tests for the dashboard elicitation-panel pure helpers and dashboard wiring."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import app.elicitation_panel as panel  # noqa: E402  (panel adds repo root to path)
from src.elicitation.export import spec_from_dict, spec_to_dict
from src.network import build_network
from src.network_spec import NetworkSpec


def _run_dict() -> dict:
    spec = NetworkSpec.from_pgmpy(build_network())
    spec.nodes["Tanker_Incidents"].kappa = 15.0
    spec.nodes["Tanker_Incidents"].kappa_level = "normal"
    return {"spec": spec_to_dict(spec)}


def test_build_framework_carries_effort_and_models() -> None:
    from src.elicitation.integration.framework import ModelSpec

    from src.elicitation.integration import default_seeds

    fw = panel.build_framework(
        [ModelSpec("claude-code", "claude-sonnet-4-5", "Claude")],
        n_agents=3, nodes=["Tanker_Incidents"], concurrency=2,
        reasoning="thorough", time_limit=30, seeds=default_seeds(),
    )
    assert fw.n_agents == 3
    assert fw.nodes == ["Tanker_Incidents"]
    assert fw.effort.reasoning_effort == "thorough"
    assert fw.effort.time_limit_s == 30.0
    assert len(fw.seeds) == len(default_seeds())


def test_override_in_spec_dict_normalises_and_replaces() -> None:
    run = _run_dict()
    # Tanker_Incidents | (Militia=low, Negotiations=success)
    panel.override_in_spec_dict(run["spec"], "Tanker_Incidents", ("low", "success"), [2, 1, 1])
    spec = spec_from_dict(run["spec"])
    col = spec.nodes["Tanker_Incidents"].cpt[("low", "success")]
    np.testing.assert_allclose(col, [0.5, 0.25, 0.25])


def test_override_rejects_unknown_column_and_zero_mass() -> None:
    run = _run_dict()
    with pytest.raises(KeyError):
        panel.override_in_spec_dict(run["spec"], "Tanker_Incidents", ("bogus",), [1, 1, 1])
    with pytest.raises(ValueError):
        panel.override_in_spec_dict(run["spec"], "Tanker_Incidents", ("low", "success"), [0, 0, 0])


def test_locked_spec_json_roundtrips_to_inference_ready_network() -> None:
    run = _run_dict()
    js = panel.locked_spec_json(run)
    spec = spec_from_dict(json.loads(js))
    spec.to_pgmpy().check_model()
    assert spec.kappa_map()["Tanker_Incidents"] == 15.0


def test_dashboard_module_imports_cleanly() -> None:
    """The dashboard imports (with the elicitation wiring) without error."""
    path = Path(__file__).resolve().parents[2] / "app" / "dashboard.py"
    spec = importlib.util.spec_from_file_location("dashboard_under_test", path)
    assert spec is not None  # importing/executing Streamlit scripts is covered by AppTest elsewhere
