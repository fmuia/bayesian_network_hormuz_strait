"""P4c — Network-view manual-override commit (AppTest).

Regression guard for the `state`-module name collision: the override panel loops
`for i, state in enumerate(states)`, which shadows a `state` module import — so
committing an override must not go through `state.<fn>`.
"""
from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _click(at, label):
    for b in at.button:
        if b.label == label:
            b.click()
            at.run()
            return True
    return False


def _app_with_node(node):
    at = AppTest.from_file("app/dashboard.py", default_timeout=90)
    at.session_state["use_fake_translator"] = True
    at.run()
    at.session_state["selected_node"] = node
    at.run()
    return at


def test_override_soft_commits_observation():
    at = _app_with_node("Tanker_Incidents")   # default sliders sum to 100 -> soft
    assert _click(at, "Set observation")
    assert not at.exception
    obs = at.session_state["observations"]
    assert len(obs) == 1 and obs[0]["soft_assignments"]


def test_override_hard_commits_observation():
    at = _app_with_node("Tanker_Incidents")
    for s, v in [("none", 100), ("isolated", 0), ("frequent", 0)]:
        at.session_state[f"soft_Tanker_Incidents_{s}"] = v
    at.run()
    assert _click(at, "Set observation")
    assert not at.exception
    obs = at.session_state["observations"]
    assert len(obs) == 1 and obs[0]["assignments"] == {"Tanker_Incidents": "none"}


def test_override_autonormalizes_non_100_sum():
    """V4 follow-up: the button is enabled at any positive sum and the committed
    soft distribution is normalised to 1.0 (no need to make sliders total 100)."""
    at = _app_with_node("Tanker_Incidents")
    for s, v in [("none", 10), ("isolated", 15), ("frequent", 25)]:   # sum 50
        at.session_state[f"soft_Tanker_Incidents_{s}"] = v
    at.run()
    assert _click(at, "Set observation")               # enabled despite sum != 100
    assert not at.exception
    soft = at.session_state["observations"][0]["soft_assignments"]["Tanker_Incidents"]
    assert soft == {"none": 0.2, "isolated": 0.3, "frequent": 0.5}
    assert abs(sum(soft.values()) - 1.0) < 1e-9


def test_posterior_panel_interval_help_is_a_tooltip_not_a_selector():
    """The interval explanation is a native hover ⓘ tooltip (markdown `help=`),
    not a popover/dropdown (which read as a control) nor a stripped HTML title."""
    at = _app_with_node("Tanker_Incidents")
    assert len(at.get("popover")) == 0          # no selector-looking chrome
    helps = [getattr(m, "help", "") or "" for m in at.markdown]
    assert any("resampling" in h for h in helps)  # tip text carried as a tooltip
    md = " ".join(m.value for m in at.markdown)
    assert "cursor:help" not in md              # the old title-bearing ⓘ span is gone


def test_scenario_node_has_no_override():
    """Scenario is inferred, not observed — the override is suppressed."""
    at = _app_with_node("Scenario")
    assert not at.exception
    assert not any(b.label == "Set observation" for b in at.button)


def test_stale_scenario_observation_does_not_crash():
    """A (legacy) hard observation on Scenario must not crash the dashboard."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
    import state  # noqa: E402

    at = AppTest.from_file("app/dashboard.py", default_timeout=90)
    at.session_state["use_fake_translator"] = True
    at.run()
    at.session_state["observations"] = [state.make_observation(
        day=1, headline="bad", assignments={"Scenario": "Severe_Closure"}, item_id="x")]
    at.run()
    assert not at.exception


def test_observed_node_panel_shows_value_and_bayes():
    """P8: selecting a hard-observed node shows its value + the standalone
    Bayes-factor contribution to the regime (not the bare flat bar)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
    import state  # noqa: E402

    at = AppTest.from_file("app/dashboard.py", default_timeout=90)
    at.session_state["use_fake_translator"] = True
    at.run()
    at.session_state["observations"] = [state.make_observation(
        day=2, headline="Fourth tanker incident this week", source="translator",
        assignments={"Tanker_Incidents": "frequent"}, item_id="x")]
    at.run()
    at.session_state["selected_node"] = "Tanker_Incidents"
    at.run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "Observed:" in md and "frequent" in md            # value + source
    assert "What this observation alone says" in md          # Bayes contribution
    assert "Bayes factor" in md


def test_soft_observed_node_shows_bayes_contribution():
    """A soft (translator ε) observation also shows the standalone Bayes-factor
    contribution — on the CI (dumbbell) panel, not just the hard-observed panel."""
    at = AppTest.from_file("app/dashboard.py", default_timeout=90)
    at.session_state["use_fake_translator"] = True
    at.run()
    at.session_state["pending_article"] = {
        "raw": "Tanker struck in the Strait of Hormuz", "body": "",
        "source": "", "source_type_label": "(unspecified — full trust)"}
    at.run()
    at.session_state["selected_node"] = "Tanker_Incidents"   # now soft-observed
    at.run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "What this observation alone says" in md
    assert "Most consistent with" in md
