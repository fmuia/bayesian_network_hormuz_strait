"""P3 (Plan 5 E3) — tests for the extracted CI / robustness chart helpers."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from components import ci_charts  # noqa: E402


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
