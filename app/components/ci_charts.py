"""CI / robustness chart helpers (Plan 5 P3 / A1b, E3).

Extracted verbatim from app/dashboard.py. ``_width_category`` and
``_ci_dataframe`` are pure; ``_dumbbell_chart`` / ``_flat_bar_chart`` build Altair
specs; ``_robustness_badge_html`` returns an HTML badge. Import-safe without a
Streamlit runtime.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import altair as alt
import pandas as pd

from theme import AMBER, GREEN, MUTED, NAVY, RED


_NAVY_FULL = NAVY
_NAVY_MID = "#5B6A7D"
_NAVY_LIGHT = "#9BA5B0"
_WIDTH_COLOR_SCALE = alt.Scale(
    domain=["narrow", "moderate", "fragile"],
    range=[_NAVY_FULL, _NAVY_MID, _NAVY_LIGHT],
)


def _width_category(half_width_pp: float) -> str:
    if half_width_pp < 8:
        return "narrow"
    if half_width_pp < 20:
        return "moderate"
    return "fragile"


# --- Continuous robustness colour (Plan 5 P6 / C2) -------------------------
# Smooth green -> amber -> red over the CI half-width (pp): the badge colour now
# shifts continuously instead of flipping at the ±8 / ±20 bucket boundaries (V3).
_ROBUST_AMBER_AT = 14.0   # half-width (pp) at full amber (centre of the old 8-20 band)
_ROBUST_RED_AT = 28.0     # half-width (pp) at/above which the colour is full red


def _lerp_hex(c1: str, c2: str, t: float) -> str:
    a = tuple(int(c1[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(c2[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % tuple(round(a[k] + (b[k] - a[k]) * t) for k in range(3))


def robustness_color(half_width_pp: float, *,
                     amber_at: float = _ROBUST_AMBER_AT,
                     red_at: float = _ROBUST_RED_AT) -> str:
    """Continuous green→amber→red hex for a CI half-width (pp). No bucket flips."""
    x = max(0.0, float(half_width_pp))
    if x <= amber_at:
        return _lerp_hex(GREEN, AMBER, x / amber_at)
    if x >= red_at:
        return RED
    return _lerp_hex(AMBER, RED, (x - amber_at) / (red_at - amber_at))


def _ci_dataframe(
    ci_dict: Dict[str, Tuple[float, float, float]],
    sorted_states: List[str],
) -> pd.DataFrame:
    rows = []
    for state in sorted_states:
        mean, lo, hi = ci_dict[state]
        half_w_pp = (hi - lo) * 50.0
        rows.append({
            "State": state,
            "Mean": mean,
            "Lo": lo,
            "Hi": hi,
            "HalfWidthPP": half_w_pp,
            "WidthCategory": _width_category(half_w_pp),
        })
    return pd.DataFrame(rows)


def _dumbbell_chart(df: pd.DataFrame, sorted_states: List[str]) -> alt.Chart:
    y_enc = alt.Y(
        "State:N", sort=sorted_states, title=None,
        scale=alt.Scale(paddingInner=0.35, paddingOuter=0.35),
        axis=alt.Axis(
            labelColor=NAVY, labelFontSize=11,
            labelOverlap=False, labelLimit=200, labelPadding=6,
        ),
    )
    x_scale = alt.Scale(domain=[0, 1])
    x_axis = alt.Axis(format="%", labelColor=NAVY, titleColor=NAVY)
    color_enc = alt.Color(
        "WidthCategory:N", scale=_WIDTH_COLOR_SCALE, legend=None,
    )
    tooltip = [
        alt.Tooltip("State:N"),
        alt.Tooltip("Mean:Q", format=".1%", title="Mean"),
        alt.Tooltip("Lo:Q", format=".1%", title="Lo (10%)"),
        alt.Tooltip("Hi:Q", format=".1%", title="Hi (90%)"),
        alt.Tooltip("HalfWidthPP:Q", format=".1f", title="± pp"),
    ]
    base = alt.Chart(df).encode(y=y_enc)
    rule = base.mark_rule(strokeWidth=4).encode(
        x=alt.X("Lo:Q", scale=x_scale, axis=x_axis, title="Probability"),
        x2="Hi:Q",
        color=color_enc,
    )
    cap_lo = base.mark_tick(thickness=3, size=18).encode(
        x="Lo:Q", color=color_enc,
    )
    cap_hi = base.mark_tick(thickness=3, size=18).encode(
        x="Hi:Q", color=color_enc,
    )
    mean_pt = base.mark_circle(size=140).encode(
        x="Mean:Q", color=color_enc, tooltip=tooltip,
    )
    height = max(160, 50 * len(sorted_states) + 30)
    return (rule + cap_lo + cap_hi + mean_pt).properties(
        height=height
    ).configure_view(stroke=None)


def _flat_bar_chart(
    dist: Dict[str, float], sorted_states: List[str]
) -> alt.Chart:
    """Plain bars without CI — for hard/soft-observed nodes."""
    df = pd.DataFrame(
        [{"State": s, "Probability": dist[s]} for s in sorted_states]
    )
    y_enc = alt.Y(
        "State:N", sort=sorted_states, title=None,
        scale=alt.Scale(paddingInner=0.35, paddingOuter=0.35),
        axis=alt.Axis(
            labelColor=NAVY, labelFontSize=11,
            labelOverlap=False, labelLimit=200, labelPadding=6,
        ),
    )
    x_enc = alt.X(
        "Probability:Q", scale=alt.Scale(domain=[0, 1]),
        axis=alt.Axis(format="%", labelColor=NAVY, titleColor=NAVY),
        title="Probability",
    )
    bars = alt.Chart(df).mark_bar(size=14, color=NAVY).encode(
        x=x_enc, y=y_enc,
        tooltip=[
            alt.Tooltip("State:N"),
            alt.Tooltip("Probability:Q", format=".1%"),
        ],
    )
    height = max(160, 50 * len(sorted_states) + 30)
    return bars.properties(height=height).configure_view(stroke=None)


def _robustness_badge_html(
    ci_dict: Dict[str, Tuple[float, float, float]],
    sorted_states: List[str],
) -> str:
    widest_state = max(
        sorted_states, key=lambda s: ci_dict[s][2] - ci_dict[s][1],
    )
    mean_w, lo_w, hi_w = ci_dict[widest_state]
    half_w_pp = (hi_w - lo_w) * 50.0
    color = robustness_color(half_w_pp)          # smooth gradient (P6 / C2 / V3)
    # The colour + the ±pp readout are the real signal; the emoji/label stay as a
    # de-emphasised coarse summary (per design decision 4).
    emoji, label = {"narrow": ("🟢", "robust"),
                    "moderate": ("🟡", "moderate"),
                    "fragile": ("🔴", "fragile")}[_width_category(half_w_pp)]
    return (
        f"<div style='font-size:0.82rem; margin:0.2rem 0 0.55rem 0; "
        f"color:{color}; font-weight:600;'>"
        f"<span style='font-size:0.78rem; opacity:0.7;'>{emoji}</span> "
        f"{label} · widest CI ±{half_w_pp:0.1f} pp "
        f"<span style='color:{MUTED}; font-weight:400;'>"
        f"(state: {widest_state})</span></div>"
    )
