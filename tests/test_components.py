"""P3/P6 (Plan 5 E3 / C2) — tests for the extracted CI / robustness chart helpers."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from components import ci_charts  # noqa: E402
from theme import AMBER, GREEN, RED  # noqa: E402


def _rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def _dist(a, b):
    return math.dist(_rgb(a), _rgb(b))


def test_width_category_boundaries():
    assert ci_charts._width_category(0) == "narrow"
    assert ci_charts._width_category(7.9) == "narrow"
    assert ci_charts._width_category(8) == "moderate"      # boundary
    assert ci_charts._width_category(19.9) == "moderate"
    assert ci_charts._width_category(20) == "fragile"      # boundary


def test_ci_dataframe_rows_and_halfwidth():
    ci = {"none": (0.5, 0.4, 0.6), "isolated": (0.3, 0.2, 0.5), "frequent": (0.2, 0.1, 0.3)}
    order = ["none", "isolated", "frequent"]
    df = ci_charts._ci_dataframe(ci, order)
    assert list(df["State"]) == order
    row = df[df["State"] == "none"].iloc[0]
    assert abs(row["HalfWidthPP"] - (0.6 - 0.4) * 50.0) < 1e-9   # 10 pp
    assert row["WidthCategory"] == "moderate"                    # 8 <= 10 < 20


def test_robustness_badge_html_categories():
    narrow = {"a": (0.5, 0.49, 0.51)}        # half-width 1 pp -> narrow
    html = ci_charts._robustness_badge_html(narrow, ["a"])
    assert "robust" in html and "🟢" in html

    fragile = {"a": (0.5, 0.2, 0.8)}         # half-width 30 pp -> fragile
    html2 = ci_charts._robustness_badge_html(fragile, ["a"])
    assert "fragile" in html2 and "🔴" in html2


def test_robustness_badge_picks_widest_state():
    ci = {"narrow_state": (0.5, 0.49, 0.51), "wide_state": (0.3, 0.1, 0.6)}
    html = ci_charts._robustness_badge_html(ci, ["narrow_state", "wide_state"])
    assert "wide_state" in html   # badge reports the widest-CI state


# ===== P6 — smooth robustness gradient (C2 / V3) ===========================


def test_robustness_color_endpoints():
    rc = ci_charts.robustness_color
    assert _rgb(rc(0)) == _rgb(GREEN)
    assert _rgb(rc(14)) == _rgb(AMBER)     # amber_at
    assert _rgb(rc(28)) == _rgb(RED)       # red_at
    assert _rgb(rc(60)) == _rgb(RED)       # clamped above red_at


def test_robustness_color_is_continuous_no_hard_flip():
    rc = ci_charts.robustness_color
    # a 2pp step straddling the old ±8 bucket boundary is a small fraction of the
    # full green→red range — i.e. a smooth shift, not a category flip.
    step = _dist(rc(7), rc(9))
    full = _dist(rc(0), rc(28))
    assert 0 < step < 0.2 * full


def test_robustness_color_monotone_within_segments():
    rc = ci_charts.robustness_color
    # green→amber leg: colour moves steadily away from green
    assert _dist(rc(3), GREEN) < _dist(rc(6), GREEN) < _dist(rc(12), GREEN)
    # amber→red leg: colour moves steadily toward red
    assert _dist(rc(16), RED) > _dist(rc(22), RED) > _dist(rc(27), RED)
