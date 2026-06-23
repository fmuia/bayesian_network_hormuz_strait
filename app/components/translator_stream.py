"""Sidebar translator subsystem (Plan 5 P4d): the news -> evidence form, the
stream callback, _run_translator (single-call or structured pipeline), source-type
resolution, and the full sidebar render. Extracted from the dashboard.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

import streamlit as st

from src.scenario import EXAMPLE_HEADLINES, TRANSLATOR_PROFILE
from src.translator import (
    SOURCE_TYPE_CREDIBILITY,
    Article,
    TranslatorError,
    TranslatorResult,
    available_providers,
    fake_forced_by_env,
    is_available as translator_available,
    structured_enabled,
    translate_article,
)
from src.translator_pipeline import run_structured
from src.ingest import fake_fetcher, ingest
from state import (
    build_review_item as _build_review_item,
    delete_named_session as _delete_named_session,
    inject_review_item as _inject_review_item,
    load_session_store as _load_session_store,
    record_observation as _append_observation,
    restore_named_session as _restore_named_session,
    save_named_session as _save_named_session,
)


STAGE_ICON = {
    "init": "🔌",
    "thinking": "💭",
    "response": "✍️",
    "parsing": "🧩",
    "validated": "✅",
}
STAGE_LABEL = {
    "init": "Connecting to model",
    "thinking": "Thinking",
    "response": "Receiving response",
    "parsing": "Parsing assignments",
    "validated": "Validated",
}


# Sidebar source-type options -> (Article.source_type, explicit credibility).
# "(unspecified)" = analyst paste at full trust (w=1.0); the rest defer to the
# per-source-type table (credibility=None -> looked up in translate_article).
_FULL_TRUST_LABEL = "(unspecified — full trust)"
_SOURCE_TYPE_OPTIONS = [_FULL_TRUST_LABEL] + list(SOURCE_TYPE_CREDIBILITY.keys())


def _resolve_source(label: str):
    """Map a sidebar source-type label to (source_type, credibility-or-None)."""
    if label == _FULL_TRUST_LABEL:
        return "unknown", 1.0
    return label, None  # None -> translate_article looks up the table


def _run_translator(article_fields: dict, stream_slot, *, provider: Optional[str] = None) -> None:
    def _write(kind: str, stage: str, detail: str) -> None:
        icon = STAGE_ICON.get(stage, "•")
        label = STAGE_LABEL.get(stage, stage.capitalize())
        clean = " ".join(detail.split())
        if len(clean) > 120:
            clean = clean[:117] + "…"
        cls = {"live": "stream-line", "done": "stream-line stream-done",
               "err":  "stream-line stream-error"}[kind]
        stream_slot.markdown(
            f"<div class='{cls}'>{icon} <b>{label}</b> — {clean}</div>",
            unsafe_allow_html=True,
        )

    _write("live", "init", "starting model call…")

    def on_step(stage: str, detail: str) -> None:
        _write("live", stage, detail)

    # Resolve the single input box (URL / bare title / manual headline+body) into
    # Article fields. A recognised news URL is fetched + split; an unlisted URL is
    # rejected. Offline/fake mode fetches saved HTML fixtures (no network).
    res = ingest(
        article_fields.get("raw", ""), article_fields.get("body", ""),
        fetcher=fake_fetcher if provider == "fake" else None,
    )
    if not res.ok:
        stream_slot.markdown(
            f"<div class='stream-line stream-error'>⛔ <b>Rejected</b> — "
            f"{res.message}</div>",
            unsafe_allow_html=True,
        )
        st.session_state.translator_error = res.message
        st.session_state.translator_raw = ""
        st.session_state.last_translation = None
        return
    if res.kind == "url":
        _write("live", "init", f"fetched from {res.site_name} ({res.source_type})")
        source_type, credibility, source = res.source_type, None, res.source
    else:
        source_type, credibility = _resolve_source(
            article_fields.get("source_type_label", _FULL_TRUST_LABEL)
        )
        source = article_fields.get("source", "")
    article = Article(
        headline=res.headline,
        body=res.body,
        source=source,
        source_type=source_type,
        url=res.url,
    )
    # T06e: when the structured toggle is on, the structured pipeline (extract →
    # map → aggregate) PRODUCES the injected assignments; otherwise the single-
    # call path does. Structured costs 2 LLM calls and derives relevance as
    # yes/no (no "partial"); the single-call path keeps the richer relevance.
    use_structured = st.session_state.get("use_structured")
    claims = mappings = None
    try:
        if use_structured:
            result, claims, mappings = run_structured(
                article, credibility=credibility, provider=provider, on_step=on_step
            )
        else:
            result = translate_article(
                article, credibility=credibility, provider=provider, on_step=on_step
            )
    except TranslatorError as exc:
        raw = getattr(exc, "raw_response", "")
        _write("err", "validated", f"failed: {exc}")
        st.session_state.translator_error = str(exc)
        st.session_state.translator_raw = raw
        st.session_state.last_translation = None
        return

    st.session_state.translator_error = None
    st.session_state.translator_raw = result.raw_response
    st.session_state.last_translation = {
        "headline": result.headline,
        "assignments": [asdict(a) for a in result.assignments],
        "rationale": result.rationale,
        "model": result.model,
        "provider": result.provider,
        "relevance": result.relevance,
    }
    if use_structured:
        st.session_state.last_translation["claims"] = [asdict(c) for c in claims]
        st.session_state.last_translation["claim_mappings"] = [asdict(m) for m in mappings]
        st.session_state.last_translation["structured_assignments"] = [
            asdict(a) for a in result.assignments
        ]

    # B3: an off-topic article abstains — logged, but no evidence injected.
    if result.relevance == "no":
        _write("done", "validated", "not relevant — no evidence injected")
        return

    if result.assignments:
        # T12: route to HITL review when flagged (partial) or when the analyst
        # has turned on "review before inject"; otherwise auto-approve (inject).
        needs_review = (
            result.relevance == "partial"
            or st.session_state.get("review_before_inject", False)
        )
        st.session_state.last_translation["pending_review"] = needs_review
        item = _build_review_item(result)
        if needs_review:
            st.session_state.review_queue.append(item)
            _write(
                "done", "validated",
                f"{len(result.assignments)} assignment(s) → queued for review "
                f"(not yet injected) · see the Triage view",
            )
        else:
            _inject_review_item(item)
            _write(
                "done", "validated",
                f"{len(result.assignments)} assignment(s) · auto-approved · model {result.model}",
            )
    else:
        st.session_state.translator_error = (
            "Translator returned no assignments — the headline does not map "
            f"to any node in this BN schema. Try a headline about {TRANSLATOR_PROFILE.domain} "
            "or use the Network tab to set a node manually."
        )
        _write("err", "validated", "no assignments produced")


# ===========================================================================
# SIDEBAR
# ===========================================================================


def render_sidebar(st):
    providers = available_providers()
    provider_labels = {"claude-code": "Claude Code", "openai": "OpenAI API"}

    # Offline `fake` translator (deterministic fixtures, no network). Default the
    # dev toggle on when TRANSLATOR_PROVIDER=fake forces it, or when no real backend
    # is available (so the app is always playable). The toggle widget owns the state.
    if "use_fake_translator" not in st.session_state:
        st.session_state.use_fake_translator = (
            fake_forced_by_env() or not translator_available()
        )
    if "use_structured" not in st.session_state:
        st.session_state.use_structured = structured_enabled()

    with st.sidebar:
        st.markdown(
            "<div class='sb-header'>"
            "<div class='sb-header-title'>Scenario Session Controls</div>"
            "<div class='sb-header-sub'>Translator, observations, and state</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # -- Fake-translator dev toggle (offline, deterministic) ---------------
        use_fake = st.toggle(
            "Use fake translator (offline)",
            key="use_fake_translator",
            help="Deterministic fixtures, no network or API key. For dev / manual "
                 "verification without spending LLM calls.",
        )
        use_structured = st.toggle(
            "Experimental: structured pipeline",
            key="use_structured",
            help="Span-grounded structured reasoning (B2): extract atomic claims → map "
                 "each to a node → aggregate. When on, this PRODUCES the injected "
                 "assignments (every one cites verbatim spans) and resists prompt "
                 "injection. Costs 2 LLM calls and derives relevance as yes/no only. "
                 "Off = the single-call path (1 call, richer relevance).",
        )
        review_before_inject = st.toggle(
            "Require review before inject",
            key="review_before_inject",
            help="Human-in-the-loop: hold every translation in the Triage view for "
                 "approve / edit / reject before it affects the model. Partial-relevance "
                 "translations are always held regardless of this toggle.",
        )
        translator_on = use_fake or translator_available()

        # -- Provider chip (one line) ------------------------------------------
        if use_fake:
            st.markdown(
                "<div class='sb-provider'>● Translator: fake (offline dev)</div>",
                unsafe_allow_html=True,
            )
        elif translator_on:
            primary = provider_labels[providers[0]]
            st.markdown(
                f"<div class='sb-provider'>● Translator: {primary}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='sb-provider warn'>⚠ No translator backend</div>",
                unsafe_allow_html=True,
            )

        _n_review = len(st.session_state.review_queue)
        if _n_review:
            st.markdown(
                f"<div class='sb-provider warn'>⏳ {_n_review} awaiting review — see Triage</div>",
                unsafe_allow_html=True,
            )

        # -- Day row ----------------------------------------------------------
        todays_count = sum(
            1 for o in st.session_state.observations
            if o["day"] == st.session_state.current_day
        )
        day_l, day_r = st.columns([1, 1], gap="small")
        with day_l:
            st.markdown(
                f"<div class='day-pill'>DAY {st.session_state.current_day}</div>"
                f"<div class='sb-hint'>• {todays_count} obs today</div>",
                unsafe_allow_html=True,
            )
        with day_r:
            if st.button("▶ Advance", width="stretch", type="secondary", key="adv_day"):
                st.session_state.current_day += 1
                st.rerun()

        st.markdown("<div class='sb-title'>Translate a URL, headline, or article</div>",
                    unsafe_allow_html=True)

        with st.form("headline_form", clear_on_submit=True):
            headline_input = st.text_area(
                "News URL or headline",
                placeholder=(
                    "Paste a news URL (Reuters, AP, BBC, Al Jazeera, …) — or just a "
                    f"headline, e.g. '{EXAMPLE_HEADLINES[0].text}'"
                    if EXAMPLE_HEADLINES else
                    "Paste a news URL (Reuters, AP, BBC, Al Jazeera, …) — or just a headline"
                ),
                height=72,
                disabled=not translator_on,
                label_visibility="collapsed",
            )
            with st.expander("Add article body & source (optional)"):
                body_input = st.text_area(
                    "Article body",
                    placeholder="Paste the article body — qualifiers in the body "
                                "(e.g. 'third such incident this week', 'no injuries') "
                                "disambiguate states the headline alone can't.",
                    height=120,
                    disabled=not translator_on,
                )
                source_input = st.text_input(
                    "Source (outlet or domain)", placeholder="e.g. Reuters",
                    disabled=not translator_on,
                )
                source_type_input = st.selectbox(
                    "Source type (sets credibility weight w)",
                    _SOURCE_TYPE_OPTIONS, index=0, disabled=not translator_on,
                )
            submitted = st.form_submit_button(
                "Translate & observe", type="primary",
                disabled=not translator_on, width="stretch",
            )
            if submitted and headline_input.strip():
                st.session_state.pending_article = {
                    "raw": headline_input.strip(),   # URL or headline; ingest resolves it
                    "body": body_input.strip(),
                    "source": source_input.strip(),
                    "source_type_label": source_type_input,
                }

        # Stream slot lives just below the form — compact, single-line.
        stream_slot = st.empty()

        # Run translator *after* the slot is in the sidebar, so updates appear here.
        if st.session_state.pending_article is not None:
            article_fields = st.session_state.pending_article
            st.session_state.pending_article = None
            _run_translator(
                article_fields, stream_slot, provider="fake" if use_fake else None
            )

        with st.expander("Examples", expanded=False):
            for idx, ex in enumerate(EXAMPLE_HEADLINES):
                if st.button(
                    ex.text, key=f"ex_{idx}",
                    width="stretch", disabled=not translator_on,
                ):
                    st.session_state.pending_article = {
                        "raw": ex.text, "body": "", "source": "",
                        "source_type_label": _FULL_TRUST_LABEL,
                    }
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='sb-title'>Named sessions</div>", unsafe_allow_html=True)
        saved_sessions = _load_session_store()
        session_name = st.text_input(
            "Session name",
            placeholder="e.g. baseline-briefing",
            key="session_name_input",
            label_visibility="collapsed",
        ).strip()

        sess_cols = st.columns([1, 1], gap="small")
        with sess_cols[0]:
            if st.button("Save session", width="stretch", key="save_named_session"):
                if not session_name:
                    st.warning("Enter a session name before saving.")
                else:
                    _save_named_session(session_name)
                    st.success(f"Saved session '{session_name}'.")
                    st.rerun()
        with sess_cols[1]:
            load_name = st.selectbox(
                "Load named session",
                options=[""] + sorted(saved_sessions.keys()),
                key="load_named_session_select",
                label_visibility="collapsed",
            )
            if st.button("Load", width="stretch", key="load_named_session"):
                if not load_name:
                    st.warning("Choose a saved session to load.")
                elif _restore_named_session(load_name):
                    st.success(f"Loaded session '{load_name}'.")
                    st.rerun()
                else:
                    st.error("Could not load that saved session.")

        if saved_sessions:
            delete_name = st.selectbox(
                "Delete named session",
                options=[""] + sorted(saved_sessions.keys()),
                key="delete_named_session_select",
                label_visibility="collapsed",
            )
            if st.button("Delete selected", width="stretch", key="delete_named_session"):
                if delete_name and _delete_named_session(delete_name):
                    st.success(f"Deleted session '{delete_name}'.")
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Reset session", width="stretch", key="reset_all"):
            st.session_state.observations = []
            st.session_state.current_day = 1
            st.session_state.last_translation = None
            st.session_state.translator_error = None
            st.session_state.translator_raw = ""
            st.session_state.selected_node = None
            st.rerun()
