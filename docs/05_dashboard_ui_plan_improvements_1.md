# Dashboard UI Plan — Improvements Round 1 (post-POC)

> **What this is.** A small post-merge UX-fix round on top of the shipped POC slice ([`05_dashboard_ui_plan.md`](05_dashboard_ui_plan.md)), driven by issues the analyst hit while actually using the app. Branch `post-p5-improvements-1` (off `main`, after the POC merged via PR #6). Companion to the POC plan and its [deferred remainder](05_dashboard_ui_plan_deferred.md).
>
> **Provenance.** Six analyst observations, 2026-06-11 (UX fixes I1–I3 below), plus a follow-on feature request in the same round — **paste-a-URL news ingestion** (see the *Feature* section). The branch also already carried `fee5db9` (a prior fix by F. Muia: override-slider layout + honest κ captions).
>
> **Status legend.** ✅ shipped (with commit) · 🅿️ deferred (parked) · ⊘ out of scope.

## The six observations and their disposition

| # | Observation | Disposition |
|---|-------------|-------------|
| 1 | Elicitation-tab node selector shows all nodes in one colour — match the graph? | ⊘ Out of scope — Streamlit limitation (below) |
| 2 | Dashboard doesn't use the full window width (big side margins) | 🅿️ Deferred → [`06`](06_dropped_to_simplify.md) §4 (`f19eb95`) |
| 3 | Override sliders should auto-normalise on apply (no manual "make it sum to 100") | ✅ **I1** (`dbb1e00`) |
| 4 | The ⓘ info icon shows nothing on hover | ✅ **I2** (`dbb1e00`) |
| 5 | Override "0/100" labels overlap text | ✅ Prior fix `fee5db9`; verified here |
| 6 | "Do a basic review for other shortcomings" | ✅ **I3** (`74cc8eb`) |

## Shipped

### ✅ I1 — Auto-normalise the override sliders *(V4 · point 3)* — `dbb1e00`
**Scope.** The manual-override sliders no longer have to sum to exactly 100. New pure helper `state.override_to_observation(vals)` normalises by the total → a sum-normalised **soft** distribution, or a **hard pin** when exactly one state is non-zero (`(None, None)` if all zero). `network_view` disables "Set observation" only when the total is 0, and shows a live `Applies as: …` preview of the normalised result.
**Closes.** The **V4** "disabled-until-100 simplex" trap — i.e. the *friction half* of **C3 / R-C3**. See reconciliation below.
**Tests.** `override_to_observation` (normalise / hard-pin / all-zero) in `test_state.py`; AppTest in `test_network_view.py` that a non-100 sum commits a normalised soft dist with the button enabled.

### ✅ I2 — Real tooltips: ⓘ popover + delta caption *(point 4)* — `dbb1e00`
**Scope.** Root cause: Streamlit's markdown sanitiser **strips the HTML `title=` attribute**, so the ⓘ hover tooltip (`network_view`) and the P9 delta-chip tooltip (`scenario_cards`) never appeared. Fixes: the ⓘ is now a real `st.popover("ⓘ about these intervals")`; the per-chip title became a single caption under the scenario cards. (Audited all `title=` uses — the only two HTML-title sites were these; every other hit is an Altair chart tooltip, which works.)
**Tests.** `test_network_view.py` asserts a popover renders and the old `cursor:help` ⓘ span is gone; `test_scenario_cards.py` asserts the caption renders.

### ✅ I3 — Escape user text in HTML surfaces + empty-source guard *(point 6, basic review)* — `74cc8eb`
**Scope.** Headlines, the translator rationale, and source were interpolated **raw** into `st.markdown(unsafe_allow_html=True)` with no escaping anywhere, so a headline containing `<`, `>`, or `&` could corrupt the layout. Apply stdlib `html.escape` at the render sites in `triage_view`, `observation_log`, and `observed_node_panel`; render the log-row source parens only when source is non-empty (was showing `()`).
**Tests.** `test_escaping.py`: a headline of `<b>boom</b> & <script>` renders escaped (`&lt;b&gt;…`), not as raw HTML.

## Feature — paste a URL or a title (news ingestion)

**What.** The translator's single input box now accepts a **news URL** (the app detects it, identifies the outlet, fetches the page, and splits it into headline + body) **or** a bare **title** (detected automatically), in addition to the unchanged **manual headline + body** path. A URL from an **unlisted / non-news** domain is rejected with a clear message and injects nothing. Everything downstream (translator, relevance abstention, HITL triage, evidence injection) is unchanged — this is a resolution layer in front of `Article` construction (`Article` already had `url`/`body`/`source`/`source_type`).

**Decision.** Curated **allow-list** of outlets (identify + reject unlisted) with lightweight BeautifulSoup + Open-Graph/JSON-LD extraction (no heavy dependency); **live HTTP fetch** at runtime **plus** saved-HTML fixtures so the fake/offline mode and the suite run with no network.

**New module `src/ingest.py`** (pure, import-safe; `requests`/`bs4` imported lazily):
- `NEWS_SITES` registry — domain → `SiteInfo(name, source_type)`. `source_type` drives the existing credibility weight `w` (`SOURCE_TYPE_CREDIBILITY`). First list (~25 outlets, one line each, extensible): wire services (Reuters, AP, AFP — w=1.0); international + regional press (BBC, Guardian, NYT, WSJ, FT, WaPo, CNN, Bloomberg, Economist, Al Jazeera, Times of Israel, Al-Monitor, Middle East Eye, The National — w=0.8); maritime trade press relevant to tanker incidents (Lloyd's List, TradeWinds, gCaptain — w=0.8); state media accepted **but heavily discounted** (Press TV, IRNA, Tasnim, RT, Xinhua — w=0.3). Al Jazeera / The National are state-funded but editorially significant → classified `commercial_press`, reclassifiable.
- `classify_input` (URL vs title — deterministic, no network), `identify_site` (`www.`/subdomain-aware), `fetch_html` (injectable `fetcher`; live `requests` default; size-capped, http(s)-only — the allow-list doubles as an SSRF guard), `extract_article` (og:title → `<title>` → `<h1>`; JSON-LD `articleBody` → `<article>` `<p>` → all `<p>`), `fake_fetcher` (offline fixtures), and `ingest(top, body, *, fetcher)` → `IngestResult`.
- **Four outcomes** of `ingest`: **manual** (a body in the expander forces the top field to be a headline) · **title** (bare headline-only) · **url-ok** (recognised outlet → fetched + split, registry `source_type`) · **url-rejected** (unlisted domain or unreadable page → `ok=False` + message).

**Wiring (`app/components/translator_stream.py`).** The top `text_area` is relabelled "News URL or headline"; submit stores `pending_article["raw"]`; `_run_translator` calls `ingest(...)` first (offline `fake_fetcher` in fake mode, else live), renders a "⛔ Rejected — …" line and returns when `not ok`, and otherwise builds the `Article` from the resolved fields (a recognised URL's registry `source_type` sets `w`, and the stream shows "fetched from {outlet}").

**Dependencies.** `requests` + `beautifulsoup4` declared in `pixi.toml [pypi-dependencies]` (already present transitively; `pixi.lock` unchanged).

**Tests (21).** `tests/test_ingest.py` (18, pure/offline): classification, site identification, extraction on a fixture + fallbacks, and all four `ingest` outcomes via stub fetchers. `tests/test_url_ingestion.py` (3, AppTest, offline via `tests/fixtures/articles/reuters.html`): a known URL fetches/splits/injects, an unlisted URL is rejected with nothing injected, a bare title still translates. Full suite **293 green**. (The four existing AppTests that simulated the form were updated `pending_article["headline"]` → `["raw"]`.)

## Deferred / not done

### 🅿️ Full-width layout *(point 2)* — recorded `f19eb95`
The dashboard caps `.block-container` at `max-width: 1600px` (`app/styles.css`), so monitors wider than ~1600px get side margins. Raising/removing the cap re-proportions **every** surface (DAG, cards, evolution chart), so it is deferred to its **own isolated layout pass** rather than riding along with these fixes. Parked in [`06_dropped_to_simplify.md`](06_dropped_to_simplify.md) §4 with rationale + trigger.

### ⊘ Elicitation-tab node colours *(point 1)*
Out of scope. Streamlit's native `selectbox`/`multiselect` render options/tags as uniformly-coloured BaseWeb elements with **no per-option colour API**, and CSS can't reliably target an individual node's tag. Colouring the picker itself would mean replacing it with a custom colored-chip control — more surface area than the value warrants now. Revisit only if a colored node picker becomes worth a custom component; the reliable fallback (a colour legend + a colored inspected-node header) remains available if wanted.

### ✅ Override "0/100" overlap *(point 5)* — prior fix verified
Already handled by `fee5db9` (state name moved to a left column; value bubble and 0/100 ticks nudged onto their own track). Verified during this round; no further change unless residual overlap is reported.

## Reconciliation with the parking-lot docs

This round changes the standing of one deferred item:

- **R-C3 / C3 (drag-to-simplex sliders).** Its acceptance scope was *(a)* remove the disabled-until-100 anti-pattern and *(b)* a drag-on-triangle interaction. **(a) is now done** (I1, auto-normalise). Updated [`05_dashboard_ui_plan_deferred.md`](05_dashboard_ui_plan_deferred.md) (R-C3 + the B3 "already addressed" note) and [`06_dropped_to_simplify.md`](06_dropped_to_simplify.md) §4 to reflect that only the optional drag interaction remains parked, with even weaker justification.

No other deferred item is touched: I2 (tooltips) and I3 (escaping) are new fixes, not items from the deferred backlog or the D-hygiene list.

## Verification

- `pixi run test` — full suite **293 green** (7 UX-fix tests + 21 ingestion tests: `test_ingest` ×18, `test_url_ingestion` ×3).
- Manual (`TRANSLATOR_PROVIDER=fake pixi run app`): non-100 slider sum commits a normalised soft observation (single slider → hard pin); the ⓘ popover opens; cards show the single ▲/▼ caption; a `<`-containing headline renders literally. **Ingestion:** paste a fixture-backed Reuters URL → headline/body auto-fill + "fetched from Reuters" + evidence injects; paste `https://example.com` → clear rejection, nothing injected; paste a plain headline → unchanged; fill the body expander → top treated as a headline.
- Commits: `dbb1e00` (I1 + I2), `74cc8eb` (I3), `f19eb95` (point-2 deferral doc); URL ingestion in a dedicated commit (below).
