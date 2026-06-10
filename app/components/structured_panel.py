"""Structured-pipeline (claims -> mappings -> aggregate) detail panel for the
latest translation (Plan 5 P4; C8 will fold it into translator_panel).
"""
from __future__ import annotations


def render(st, t):
    if "claims" in t:
        _claims = t["claims"]
        _maps = t.get("claim_mappings", [])
        _by_span = {m["supporting_span"]: m for m in _maps if m.get("supporting_span")}
        with st.expander(
            f"Structured pipeline (experimental) — {len(_claims)} claim(s), "
            f"{len(_maps)} mapped",
            expanded=False,
        ):
            st.caption(
                "Span-grounded atomic claims (B2 step 1) mapped to BN nodes "
                "(step 2) then aggregated (step 3). Each claim cites a verbatim "
                "span copied from the article; ungrounded claims are dropped. "
                "The aggregated output below **is** what was injected."
            )
            if t.get("claims_error"):
                st.warning(f"Structured pipeline failed: {t['claims_error']}")
            for c in _claims:
                span = c["verbatim_span"]
                m = _by_span.get(span)
                if m:
                    mapped = (
                        f" → **{m['node'].replace('_',' ')} = {m['state']}**"
                    )
                else:
                    mapped = " → <span class='muted-note'>(no node)</span>"
                st.markdown(f"- “{span}”{mapped}", unsafe_allow_html=True)
            if not _claims and not t.get("claims_error"):
                st.markdown("_No grounded claims extracted._")
            _agg = t.get("structured_assignments")
            if _agg is not None:
                st.markdown("**Aggregated pipeline output (injected):**")
                if _agg:
                    for a in _agg:
                        st.markdown(
                            f"- {a['node'].replace('_',' ')} = **{a['state']}**"
                        )
                else:
                    st.markdown("_No nodes mapped — abstained._")
