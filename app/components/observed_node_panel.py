"""Rich panel for a hard-observed node (Plan 5 P8 / C4 / V5).

Replaces the uninformative flat 100%/0% bar with: the observed value + its source
and day, the (kept) flat bar, and — first-class via Plan 1 — this observation's
*standalone* Bayes-factor contribution to the latent regime, i.e. how strongly
this single observation favours one Scenario state over another, independent of
the prior. The Bayes block is omitted on non-latent topologies (``bayes`` is
``None``).
"""
from __future__ import annotations

from math import inf

from components.ci_charts import _flat_bar_chart
from theme import MUTED, NAVY, SCENARIO_COLOR, SCENARIO_KEYS, SCENARIO_LABEL


def _bayes_html(bayes: dict) -> str:
    """A rel-likelihood bar per regime + the headline pairwise Bayes factor."""
    rel = {s: v for s, v in bayes.get("rel_like", {}).items() if v != inf}
    if not rel:
        return ""
    top = max(rel, key=rel.get)
    bot = min(rel, key=rel.get)
    factor = bayes.get("lambda", {}).get(top, {}).get(bot)

    rows = ""
    for s in SCENARIO_KEYS:
        v = float(rel.get(s, 0.0))
        rows += (
            f"<div style='display:flex; align-items:center; gap:0.4rem; margin:0.12rem 0;'>"
            f"<div style='width:7.4rem; font-size:0.72rem; color:{NAVY};'>{SCENARIO_LABEL[s]}</div>"
            f"<div style='flex:1; background:#EEF0F2; border-radius:3px; height:0.6rem;'>"
            f"<div style='width:{v * 100:.0f}%; background:{SCENARIO_COLOR[s]}; "
            f"height:100%; border-radius:3px;'></div></div>"
            f"<div style='width:2.4rem; font-size:0.72rem; text-align:right; "
            f"color:{MUTED};'>{v:.2f}</div></div>"
        )
    head = f"Most consistent with <b style='color:{SCENARIO_COLOR[top]}'>{SCENARIO_LABEL[top]}</b>"
    if factor not in (None, inf) and factor > 1.0:
        head += f" — {factor:.1f}× more likely than {SCENARIO_LABEL[bot]} (Bayes factor)"
    return (
        f"<div style='font-size:0.74rem; color:{MUTED}; margin:0.5rem 0 0.15rem;'>"
        f"What this observation alone says about the regime:</div>"
        f"{rows}"
        f"<div style='font-size:0.74rem; color:{NAVY}; margin-top:0.25rem;'>{head}</div>"
    )


def render_bayes_contribution(st, bayes) -> None:
    """Render the standalone Bayes-factor block. Shared by the hard-observed panel
    (below) and the soft-observed CI panel in network_view. No-op if ``bayes`` is
    falsy (e.g. a non-latent topology or an unobserved node)."""
    html = _bayes_html(bayes) if bayes else ""
    if html:
        st.markdown(html, unsafe_allow_html=True)


def render(st, *, observed_state, meta, bayes, marginal, sorted_states):
    src = meta.get("source", "?")
    day = meta.get("day")
    headline = meta.get("headline", "")
    day_txt = f" · day {day}" if day is not None else ""
    st.markdown(
        f"<div class='card-sub'>Observed: <b>{observed_state}</b> "
        f"<span style='color:{MUTED};'>({src}{day_txt})</span></div>"
        + (f"<div class='meta' style='margin:-0.15rem 0 0.3rem;'>“{headline}”</div>"
           if headline else ""),
        unsafe_allow_html=True,
    )
    st.altair_chart(_flat_bar_chart(marginal, sorted_states), width="stretch")
    render_bayes_contribution(st, bayes)
