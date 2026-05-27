# Dashboard UI, Visualization, and Polish Plan

> **Status.** Draft. No items started.
>
> **Position in the sequencing.** Last (fourth) of the four sequential engineering plans; fifth overall in the programme (Plan 1 is the latent-regime reframing — conceptual decision plus engineering — that the engineering plans build on; Plans 2–4 are the upstream engineering plans). Closes the visualization, dashboard-architecture, performance, code-hygiene, and test-coverage findings (catalogued in `docs/master_plan.md` §4) that are not addressed by the three backend plans. Runs after Plans 2–4, although several items in Category A and D can be opportunistically picked up earlier.
>
> **Related docs.** `docs/master_plan.md` §4 is the in-tree registry of finding IDs and lists the findings this plan closes (V1–V15, C1–C14 where not addressed elsewhere, M8, M9). `docs/bn_app_next_steps.md` is the feature roadmap (A1–E1) — several roadmap items are UI-flavoured (A3 sensitivity attribution, B2 scenario sequences, C1 comparison, C3 undo/redo) and naturally compound with this plan's deliverables.
>
> **Status legend.** ⬜ not started · ⏳ in progress · ✅ shipped (with date).

## Executive Summary

The three backend plans (translator robustification, PyMC migration, elicitation tool) address evidence ingestion, inference, and methodology. They leave a clean coverage gap: **dashboard architecture, visualization, performance, code hygiene, and test coverage**. This plan closes that gap.

The plan is organised into five categories:

- **Category A — Architectural refactor.** The 1878-line `app/dashboard.py` is split into a component library; the inline CSS is extracted; the cached-but-mutated `engine` pattern is fixed; the hardcoded DAG layout becomes topologically derived.
- **Category B — Performance and caching.** Streamlit caches gain bounds and TTLs; the probability-evolution series is memoised on observation IDs rather than evidence values; slider interactions stop triggering full recomputation.
- **Category C — Visualization improvements.** Fourteen discrete improvements to the user-facing experience: better scenario card density, smoother robustness gradient, drag-to-simplex sliders, richer panels for observed nodes, improved DAG layout, distinguished visual encoding for parameter vs forecast uncertainty, before/after deltas on new observations, stacked bar visualization of translator distributions, colorblind-safe palette, multi-line tooltips, edge rationale on hover, responsive scenario narratives, flex/grid DAG canvas, and a continuous Oil_Price panel with interval queries (paired with Plan 3 Phase 3/4).
- **Category D — Code hygiene.** Dead code removal, unused field cleanup, defensive-guard documentation, and deduplication of sensitivity functions.
- **Category E — Test coverage.** Tests for the helpers and visualisation primitives the current test suite doesn't reach.

Categories A and D are pure refactor / cleanup. Category B is performance optimisation. Category C is the user-facing polish that converts the demo into something usable in committee settings. Category E is hygiene.

## Context

### Position in the broader plan stack

Four plans run sequentially:

1. **Plan 2 — `docs/02_translator_robustification.md`** — evidence ingestion.
2. **Plan 3 — `docs/03_pymc_integration_plan.md`** — inference engine.
3. **Plan 4 — `docs/04_elicitation_tool_plan.md`** — methodology layer.
4. **Plan 5 (this doc)** — dashboard UI, visualization, performance, polish.

The backend plans deliver capabilities; Plan 5 delivers the user-facing experience that exposes those capabilities. Dependencies are item-specific rather than category-wide:

- **A3 (engine caching fix)** lands cleanly once Plan 3 Phase 1's `Posterior` interface is in place — without it, A3 still works but the "pure-function query" target shape isn't established. A3 can also ship pgmpy-only as an interim.
- **C4 (rich observed-node panel)** has a Bayes-factor mode that requires Plan 1 (latent regime). Falls back to delta display if Plan 1's engineering hasn't landed (see open question in Section C).
- **C14 (continuous Oil_Price panel)** requires Plan 3 Phase 3 (continuous-node support in `PymcBackend`); production data wiring lands with Plan 3 Phase 4.
- **C8 (stacked-bar translator distributions)** can ship on the current translator output; richer per-sample views compound with Plan 2 C1 (ensemble) once it lands.

The rest of Plan 5 — A1/A2/A4, B1/B2/B3, the bulk of Category C, and all of D and E — is backend-independent and can be picked up opportunistically alongside the other plans.

### Relationship to the existing feature roadmap

`docs/bn_app_next_steps.md` contains a feature roadmap with items A1–E1. Several of those items are UI-flavoured and compound with this plan's work:

- **Roadmap A3 (sensitivity attribution)** — compounds with this plan's C7 (before/after delta on new observation). Plan 5's delta-chip is the minimum-viable version; A3 is the full waterfall attribution.
- **Roadmap B2 (pre-built scenario sequences)** — demo-time UX; compounds with Plan 5's general UI polish.
- **Roadmap C1 (scenario comparison mode)** — multi-pane forked-evidence view. Compounds with Plan 5's information-architecture work in C1.
- **Roadmap C3 (undo/redo and pinning)** — UI feature, compounds with Plan 5's evidence-management UX.
- **Roadmap D1 (batch processing)** — UI for multi-headline ingest, compounds with Plan 2's batch capabilities.

This plan does not duplicate those items; they remain in the roadmap doc. The orchestrator (`docs/master_plan.md`) maps the coupling explicitly.

## Diagnosis: What's Wrong With the Current Dashboard

The findings list this plan closes. Items marked (V*/C*/M*) are finding IDs from the master-plan §4 matrix.

1. **`app/dashboard.py` is 1878 lines / 76 KB in a single file (C2).** CSS, session state, four tabs of UI, helper components, and computation are interleaved. Any change risks regressing unrelated parts; nothing in the helpers is unit-testable.
2. **The CSS is embedded inline (~350 lines) (V8).** Style edits require scrolling past hundreds of `!important` rules to find the Python code.
3. **The `engine` is `@st.cache_resource` but mutated on every rerun (C1).** Cross-session bleed and race conditions in any deployment beyond a single laptop user.
4. **The DAG layout (`_NODE_LEVEL`) is hardcoded (C10).** If a node is added to the network, the layout silently defaults to level 0.
5. **The probability-evolution chart recomputes the full posterior series on every rerun (C3).** Slider tweaks bust the cache keys and trigger visible recomputation.
6. **Streamlit caches have no bounds (C4).** Memory leak in long-running sessions.
7. **Scenario cards eat ~1/3 of the viewport for three numbers (V2).** Information density is wrong; the evolution chart is the story but is comparatively small.
8. **Robustness thresholds (±8, ±20 pp) are arbitrary, with sharp colour transitions (V3).** A node with 7.9pp half-width is 🟢; 8.1pp is 🟡. Stakeholders read meaning into transitions the thresholds don't support.
9. **Manual override sliders are a UX trap (V4).** Three independent 0-100 sliders, primary button disabled until they sum to exactly 100. Friction reduces exploratory use.
10. **Hard-observed nodes show an uninformative bar chart (V5).** A bar at 100% on one state, 0% on others, in plain navy — the panel for the most interesting nodes is the least informative.
11. **DAG layout: level 3 stacks four nodes, level 4 has only Scenario (V6).** Visual clutter on the most-watched part of the graph.
12. **Probability-evolution band conflates parameter uncertainty with forecast uncertainty (V7).** The shaded band looks like a forecast confidence band but actually measures model robustness; the caption calls this out but the visual is misleading regardless.
13. **No before/after visualization on new observations (V9).** Stakeholders mentally subtract two numbers; no delta, no animation, no callout.
14. **Latest-translation panel renders soft-evidence distributions as plain text (V10).** Harder to scan than a stacked bar.
15. **Green/red palette is colourblind-hostile (V11).** Stress vs Severe collapses under deuteranopia.
16. **Headline truncation on evolution-chart tooltip is aggressive (V12).** 180-char cap clips most of the second headline.
17. **Edge rationale is isolated from the graph (V13).** Rationale lives in a separate tab; the DAG can't surface it on hover.
18. **Scenario cards have fixed narrative paragraphs (V14).** Same text on Day 0 and Day 30; decoration after two demos.
19. **460px DAG canvas height is calibrated to the right column (V15).** Brittle layout that breaks on small UI changes elsewhere.
20. **`render_network_png` is dead code (C9).** ~60 lines of HTML-label code that no one uses.
21. **`_PLUGINS_REGISTERED` workaround needs a TODO (C12).** Subprocess workaround for a conda-forge packaging bug, lacks reminder to remove.
22. **`Observation.tone` is declared but unused (C13).** Type vocabulary noise.
23. **No tests for `_merged_evidence` (C11).** Soft↔hard evidence ordering invariants untested; easy to regress.
24. **No tests for `viz.py` or dashboard helpers (C14).** The visualization and state logic most likely to silently regress on a refactor.
25. **`+1e-6` alpha guard in `_resample_cpd` never fires meaningfully (M8).** Unexplained constant that could mask a real bug if a future CPT edit introduces structural zeros.
26. **`scenario_credible_intervals` and `node_credible_intervals` duplicate the resampling loop (M9).** Two implementations to keep in sync.

## Section A — Layered plan

Each item has a clear scope, deliverable, and validation criterion.

## Category A — Architectural refactor

These items are pure refactoring with no semantic change. They establish the structural foundation that the rest of the plan builds on.

### A1. Split `app/dashboard.py` into a component library

**Status.** ⬜ not started
**Resolves.** C2 from the review.

**Scope.** Decompose the 1878-line monolith into:

- `app/dashboard.py` — orchestration only. Page setup, session state, tab routing.
- `app/styles.css` — extracted CSS (see A2).
- `app/state.py` — session-state defaults, named-session save/load, `_merged_evidence`, `_append_observation`.
- `app/components/scenario_cards.py` — pinned-band scenario cards.
- `app/components/ci_charts.py` — dumbbell chart, CI dataframe helpers, robustness badge.
- `app/components/network_tab.py` — Network tab: agraph + override + posterior panel.
- `app/components/observation_log.py` — Observations tab and audit log.
- `app/components/edge_rationale.py` — Edge rationale tab.
- `app/components/audit_tab.py` — Audit trail tab.
- `app/components/translator_stream.py` — sidebar translator-stream UI.

Each component module exposes one or a few public render functions consuming a `Posterior` (from Plan 3) and session state. Pure rendering; no inference, no global state mutation.

**Anticipated additional components.** Items downstream of A1 introduce new component files. A1 establishes the directory and conventions; later items add to it without restructuring. The expected additions are:

- `app/components/override_panel.py` — added in C3 (drag-to-simplex sliders).
- `app/components/observed_node_panel.py` — added in C4 (rich panel for hard-observed nodes).
- `app/components/evolution_chart.py` — split out as C6 / C10 land (band rendering + tooltips).
- `app/components/translator_panel.py` — added in C8 (stacked-bar distributions). Distinct from `translator_stream.py`: the stream is the sidebar feed; the panel is the per-observation detail view.
- `app/components/sources_tab.py` — Plan 2 B1b's Sources tab, shared with Plan 4 Layer 3.
- `app/components/continuous_viz.py` — added in C14 (continuous Oil_Price panel), once Plan 3 Phase 3 lands.

**Deliverables.**

- `app/components/` directory with the modules listed above.
- `app/state.py`, `app/styles.css`.
- `app/dashboard.py` reduced to orchestration.

**Validation.**

- Visual parity with the current dashboard (screenshot diff on the canonical evidence states).
- Each helper module independently importable in a unit test (no Streamlit context needed).
- Line count in `app/dashboard.py` drops below ~250 lines.

### A2. Extract CSS to a separate stylesheet

**Status.** ⬜ not started
**Resolves.** V8 from the review.

**Scope.** Move the ~350 lines of inline CSS in `app/dashboard.py` lines 88–443 to `app/styles.css`. Load via `st.markdown(open(...).read(), unsafe_allow_html=True)` at app start. Organise sections within the CSS file with comments matching the current inline structure (sidebar toggle, scenario cards, sliders, translator stream, etc.).

**Deliverables.**

- `app/styles.css`.
- `app/dashboard.py` loads it at startup; the inline `<style>` block is removed.

**Validation.**

- Visual parity with the current dashboard.
- CSS is editable independently; reloading the page picks up changes without Python edits.

### A3. Fix the cached-but-mutated `engine` pattern

**Status.** ⬜ not started
**Resolves.** C1 from the review.

**Scope.** The current pattern (`engine = get_engine(); engine.clear_evidence(); engine.update_evidence(...)`) mutates a `cache_resource`-cached object on every rerun, causing cross-session bleed.

Two acceptable resolutions:

1. **Pure-function queries.** `query(net, evidence)` returns a `Posterior` without mutating any cached object. The `net` is cached as a `cache_resource`; the engine is constructed per-query (cheap operation).
2. **Per-session engine.** Move the engine into `st.session_state` so each user gets their own.

Option 1 aligns with the backend interface in Plan 3 (Posterior consumer pattern). Recommend Option 1.

**Deliverables.**

- `src/inference.py` updated to expose pure-function query API.
- `app/dashboard.py` and `app/state.py` updated to consume it.
- Same pattern applied to the probability-evolution loop.

**Validation.**

- Two parallel sessions (two browser tabs) maintain independent evidence states.
- The cached `network` is the only shared object; mutating evidence in one session does not affect another.
- Existing dashboard behaviour preserved.

### A4. Derive DAG layout levels topologically

**Status.** ⬜ not started
**Resolves.** C10 from the review.

**Scope.** Replace the hardcoded `_NODE_LEVEL` dict in `src/viz.py:229-243` with a function that computes per-node levels from the DAG (longest path from any root). Memoize the result for cheap reuse. Add a fallback that places nodes not in the computed layout at level 0 with a warning logged.

**Deliverables.**

- `src/viz.py` — `compute_node_levels(spec: NetworkSpec) -> dict[str, int]` replacing `_NODE_LEVEL`.
- Existing layout produces identical levels on the current network (validation).

**Validation.**

- `compute_node_levels(build_hormuz_spec()) == _NODE_LEVEL` on the current network.
- Adding a new node to `STATES` correctly assigns it a level without manual editing.

## Category B — Performance and caching

These items reduce computational waste in the interactive dashboard. None affect output correctness.

### B1. Bound Streamlit caches

**Status.** ⬜ not started
**Resolves.** C4 from the review.

**Scope.** Add `max_entries` and `ttl` to `cached_credible_intervals` and `cached_node_credible_intervals` in `app/dashboard.py:456-474`. Defaults: `max_entries=64, ttl="1h"`.

**Deliverables.**

- Decorators updated.
- A short comment explaining the rationale.

**Validation.**

- Memory usage is bounded across a long session of slider tweaks (manual test or process-monitor smoke check).

### B2. Memoise the probability-evolution series on observation IDs

**Status.** ⬜ not started
**Resolves.** C3 from the review.

**Scope.** The probability-evolution chart currently rebuilds the full posterior series on every rerun. Replace with a memoisation keyed on the sorted tuple of observation IDs:

- `@st.cache_data` keyed on `tuple(sorted(obs["id"] for obs in observations))`.
- The cached function returns the long-form DataFrame.
- New observations invalidate the key naturally; slider tweaks on overrides don't (they modify the in-progress observation but don't append to the committed list).

**Deliverables.**

- Refactored evolution-chart computation in `app/components/scenario_cards.py` (post-A1).

**Validation.**

- Cache hit rate observed via debug logging on a sequence of slider tweaks vs new-observation commits.
- Visual parity with the current chart.

### B3. Reduce recomputation on slider tweaks

**Status.** ⬜ not started
**Resolves.** Part of C3; UX latency.

**Scope.** The manual-override panel's sliders trigger rerenders of every chart. Two patterns mitigate this:

1. **Debounce slider changes.** Use a callback that only commits to session state after a short idle (Streamlit's `on_change` + a debounce helper, or manual flag).
2. **Hold-mode rendering.** Show the override-pane numbers updating live but defer the downstream chart re-render until the user clicks "Apply."

Recommend the second pattern: clearer mental model for the user, no debounce-tuning headaches.

**Deliverables.**

- `app/components/network_tab.py` — override panel updated with explicit "Apply" / "Reset" buttons.

**Validation.**

- Slider interactions feel responsive (no chart re-render on each tick).
- Apply button correctly commits the override as a soft-evidence observation.

## Category C — Visualization improvements

User-facing polish. These items directly address V findings from the review.

### C1. Right-size scenario cards, expand the evolution chart

**Status.** ⬜ not started
**Resolves.** V2 from the review.

**Scope.** The three scenario cards currently consume ~1/3 of the viewport for three numbers. Compress to a vertical strip (~1/5 viewport) and reallocate the saved space to the probability-evolution chart, which is the only thing that moves with new evidence and therefore the actual locus of attention.

**Deliverables.**

- `app/components/scenario_cards.py` — compact strip layout.
- Updated grid CSS in `app/styles.css`.

**Validation.**

- Visual comparison: evolution chart is at least 1.5× larger than current; cards remain legible at all common breakpoints.

### C2. Smooth robustness gradient instead of three-bucket emoji

**Status.** ⬜ not started
**Resolves.** V3 from the review.

**Scope.** Replace the three-bucket 🟢🟡🔴 categorisation (current thresholds: ±8, ±20 pp) with a smooth gradient. Two options:

1. **LERP between green/amber/red along the half-width.** Continuous colour mapping; no transitions.
2. **Calibrated thresholds against the actual network.** Define "fragile" as the worst quartile of half-widths under the prior.

Recommend Option 1 for the badge colour, with a small numerical readout next to it. Optionally retain emoji as a coarse summary for quick reads, but de-emphasise it.

**Deliverables.**

- `app/components/ci_charts.py` — `robustness_badge` function refactored.

**Validation.**

- A node moving from half-width 7pp to 9pp shows a smooth colour shift rather than a category flip.
- The badge still communicates "robust" / "moderate" / "fragile" at a glance.

### C3. Drag-to-simplex override sliders

**Status.** ⬜ not started
**Resolves.** V4 from the review.

**Scope.** Replace the three-independent-sliders pattern with one of:

1. **Drag-to-simplex.** Moving one slider auto-rescales the others proportionally so the total stays at 100.
2. **Anchor + distribute.** User enters one state's value; the others auto-fill proportionally to the current prior or a uniform fallback.

Recommend implementing both as toggleable modes; analysts have preferences.

**Deliverables.**

- `app/components/override_panel.py` — new component with the two modes.
- Removes the disabled-until-100 button anti-pattern.

**Validation.**

- Manual interaction tests; sliders feel natural; total always 100.
- Soft evidence committed correctly to session state.

### C4. Rich panel for hard-observed nodes

**Status.** ⬜ not started
**Resolves.** V5 from the review.

**Scope.** Replace the flat 100%/0% bar for hard-observed nodes with an information-rich panel:

- Observed value with source (headline or manual override).
- Day observed.
- Downstream effect: change in scenario posterior since the observation was committed.
- The previous posterior (before observation) for comparison.

If the latent regime (Plan 1) is in place, also show the Bayes-factor contribution of this observation.

**Deliverables.**

- `app/components/observed_node_panel.py`.

**Validation.**

- Visual comparison of the panel before and after on the canonical Hormuz escalation scenario.

### C5. DAG layout improvement

**Status.** ⬜ not started
**Resolves.** V6 from the review.

**Scope.** Move `Oil_Price_Regime` to its own level (parallel to `Scenario`, not before it). Once A4 (topological derivation) is in place, this becomes automatic — `Oil_Price_Regime` has no outgoing edge to `Scenario`, so it sits at the same level. Until A4 ships, the manual `_NODE_LEVEL` entry should be updated.

**Deliverables.**

- Updated layout in `src/viz.py`.

**Validation.**

- Visual comparison: right side of the graph less crowded.
- Level-3 row contains three nodes rather than four; level-4 contains both `Scenario` and `Oil_Price_Regime`.

### C6. Distinguish parameter uncertainty from forecast uncertainty visually

**Status.** ⬜ not started
**Resolves.** V7 from the review.

**Scope.** The probability-evolution band is "80% CI on the posterior at that day's evidence state" — model robustness, not a forecast. Render it visually distinct from what a forecast band would look like:

- Hashed fill or dotted edges.
- Anchored annotation: "Band = robustness; not a forecast."
- A true forecast band (regime-evolution uncertainty over time) requires a temporal BN extension that is out of scope for the four plans (see `docs/master_plan.md` §6 and `docs/bn_hmm_integration.md`). If/when it ships, render it in a contrasting style.

**Deliverables.**

- `app/components/evolution_chart.py` updated band rendering.

**Validation.**

- Visual comparison; analyst (or test stakeholder) reads the band correctly without the caption.

### C7. Before/after delta on new observation

**Status.** ⬜ not started
**Resolves.** V9 from the review.

**Scope.** When a new observation is committed, display:

- A chip on each scenario card: `▲ +5pp Severe` or `▼ -3pp Stress`.
- A brief animation or visual cue indicating "this changed because of the most recent observation."

This is the minimum-viable version of roadmap A3 (sensitivity attribution). The full A3 waterfall is a richer follow-on; this delta-chip works on its own.

**Deliverables.**

- `app/components/scenario_cards.py` — delta chip overlay.
- `app/state.py` — store previous-observation scenario probabilities for the diff.

**Validation.**

- Visual check: committing a new observation produces visible deltas on the affected scenarios.
- Edge cases: undo / remove observation restores the previous state cleanly.

### C8. Stacked-bar visualization of translator distributions

**Status.** ⬜ not started
**Resolves.** V10 from the review.

**Scope.** In the "Latest translation" panel, render each per-assignment `state_probs` distribution as an inline stacked bar (~60px wide) instead of plain text. Tooltip retains the text representation for accessibility.

**Deliverables.**

- `app/components/translator_panel.py` — new render with stacked bar.

**Validation.**

- Visual scan of the panel is faster (informal user test); text values still accessible on hover.

### C9. Colorblind-safe palette

**Status.** ⬜ not started
**Resolves.** V11 from the review.

**Scope.** Replace the current scenario palette (GREEN `#2E8B57`, AMBER `#D4A017`, RED `#B22222`) with a CVD-safe alternative:

- Recommended palette: `#0072B2` (Stress), `#E69F00` (Prolonged), `#D55E00` (Severe). Wong's 2011 CVD-safe set.
- Also add shape/style encoding on chart lines (solid / dashed / dotted) so the distinction survives even pure-greyscale rendering.

**Deliverables.**

- `app/styles.css` — palette constants updated.
- All Altair charts updated to use the new colour scale.
- Line styles added to the evolution chart.

**Validation.**

- CVD simulator (Coblis or similar) shows the three scenarios remain distinguishable.
- Stakeholder sample review with a colour-aware designer.

### C10. Multi-line tooltips on the evolution chart

**Status.** ⬜ not started
**Resolves.** V12 from the review.

**Scope.** Remove the 180-character truncation on the headline tooltip in the evolution chart. List each headline on its own line within the tooltip; Altair tooltips handle multi-line content correctly.

**Deliverables.**

- `app/components/evolution_chart.py` updated tooltip configuration.

**Validation.**

- Visual check: a day with three headlines shows all three in full in the tooltip.

### C11. Edge rationale on hover in the DAG

**Status.** ⬜ not started
**Resolves.** V13 from the review.

**Scope.** The edge-rationale text in the Edge Rationale tab is currently disconnected from the DAG. Surface each edge's rationale as a tooltip in the agraph view:

- `streamlit_agraph` supports edge titles; populate them from `_EDGE_RATIONALE`.
- Hovering an edge shows the rationale text.
- The Edge Rationale tab remains for full-list reading and for the omitted-edges discussion.

**Deliverables.**

- `src/viz.py` — `build_agraph_payload` populates edge titles.

**Validation.**

- Hover test on the canonical Hormuz DAG: rationale text appears for every edge.

### C12. Responsive scenario narratives

**Status.** ⬜ not started
**Resolves.** V14 from the review.

**Scope.** The fixed narrative paragraph on each scenario card doesn't change with evidence — decoration after two demos. Two options:

1. **Remove from always-on card.** Move to a hover or expander; the always-on card shows only the probability and the CI.
2. **Make narratives responsive.** Requires roadmap B1 (daily narrative generation); generate a short paragraph per scenario per day. Compounds with B1 from `bn_app_next_steps.md`.

Recommend Option 1 in the interim; Option 2 lands when B1 ships.

**Deliverables.**

- `app/components/scenario_cards.py` — narrative moved to an expander.

**Validation.**

- Cards visually denser; narrative remains accessible on click.

### C13. Flex/grid layout for DAG canvas

**Status.** ⬜ not started
**Resolves.** V15 from the review.

**Scope.** The 460px DAG canvas height in `src/viz.py:411-412` is hardcoded to match the right-column content. Replace with a flexbox or grid layout that adapts the canvas height to the right column's actual rendered height.

Streamlit's layout primitives are limited; implementation likely needs a small custom component or JavaScript bridge. Alternative: accept the brittleness for now and re-evaluate when migrating off Streamlit.

**Deliverables.**

- Either: a flex/grid layout implementation.
- Or: a documented decision to defer until UI framework migration, with a TODO at the hardcoded value.

**Validation.**

- If implemented: DAG canvas matches right-column height on common viewport sizes.

### C14. Continuous Oil_Price panel and interval queries

**Status.** ⬜ not started
**Resolves.** Surfaces Plan 3 Phase 3/4 outputs in the UI. No specific V-finding — added so the continuous-variable visualization work has a clear owner in Plan 5 rather than being smuggled into Plan 3.

**Scope.** When Plan 3 Phase 3 ships continuous-node support and Phase 4 promotes `Oil_Price` to a continuous LogNormal, the dashboard needs:

- A density-plot panel for the continuous Oil_Price posterior (replacing the 3-state bar chart).
- Interval-probability readouts on the scenario cards: $P(\text{Oil} > 120 \mid E)$, $P(\text{Oil} \in [100, 140] \mid E)$, plus an analyst-editable threshold.
- A small "interval query" widget that takes a user-supplied range and returns the posterior mass over it.

Lands behind a capability flag from the `Posterior` object — only renders when the backend reports continuous Oil_Price; falls back to the existing bar chart on the discrete backend.

**Deliverables.**

- `app/components/continuous_viz.py` — density plot, interval-query widget. The corresponding `Posterior.probability_of_interval` / `Posterior.density` API is shipped by Plan 3 Phase 3.
- `app/components/scenario_cards.py` — wire interval-probability readout into the card when continuous Oil_Price is present.

**Validation.**

- Discrete backend: existing bar-chart behaviour preserved (regression test on the canonical Hormuz spec).
- Continuous backend: density plot renders for a Phase-4 test spec; interval queries return values consistent with `Posterior.probability_of_interval` direct calls.

## Category D — Code hygiene

Small individual items that accumulate. Mostly mechanical.

### D1. Remove dead code: `render_network_png`

**Status.** ⬜ not started
**Resolves.** C9 from the review.

**Scope.** `render_network_png` in `src/viz.py:156-211` is unused by the dashboard (which uses `build_agraph_payload`). Two options:

1. Delete it now.
2. Keep it as the natural implementation for roadmap B3 (session export to PDF / standalone HTML).

Recommend Option 2 with a `# Used by: docs/bn_app_next_steps.md B3 — session export` comment so it's not deleted accidentally.

**Deliverables.**

- Comment added, or function deleted.

**Validation.**

- Test coverage either confirms deletion or documents intended use.

### D2. Drop unused `Observation.tone`

**Status.** ⬜ not started
**Resolves.** C13 from the review.

**Scope.** `Tone` literal in `src/evidence.py:15, 33-36` is used only by `ExampleHeadline`. Either:

1. Thread tone through `Observation` (could feed into roadmap B1 narrative generation).
2. Drop the field from the type vocabulary.

Recommend Option 2 unless B1 is being shipped at the same time.

**Deliverables.**

- `src/evidence.py` — `Tone` removed or threaded.

**Validation.**

- Tests pass; no callers broken.

### D3. Clean up or document the `+1e-6` Dirichlet alpha guard

**Status.** ⬜ not started
**Resolves.** M8 from the review.

**Scope.** The `alpha = concentration * values[:, col] + 1e-6` in `src/sensitivity.py:27` is a defensive guard against zero alpha that never fires under current CPT values (minimum is 0.01).

Two options:

1. Remove for clarity.
2. Keep with a comment explaining the defensive intent against future CPT edits introducing structural zeros.

Recommend Option 2: low cost, marginal safety value.

**Deliverables.**

- `src/sensitivity.py` updated with clarifying comment.

**Validation.**

- Tests pass.

### D4. Deduplicate `scenario_credible_intervals` and `node_credible_intervals`

**Status.** ⬜ not started
**Resolves.** M9 from the review.

**Scope.** The two functions in `src/sensitivity.py:48-84` and `:104-181` implement essentially the same Monte-Carlo procedure twice. Have `scenario_credible_intervals` forward to `node_credible_intervals(..., nodes=["Scenario"])`, or delete it entirely and update callers.

**Deliverables.**

- `src/sensitivity.py` — `scenario_credible_intervals` deleted or refactored as a thin wrapper.
- Callers updated.

**Validation.**

- `tests/test_sensitivity.py:test_node_ci_matches_scenario_ci_for_scenario_node` continues to pass.

### D5. TODO comment on `_PLUGINS_REGISTERED` workaround

**Status.** ⬜ not started
**Resolves.** C12 from the review.

**Scope.** The `subprocess.run(["dot", "-c"])` workaround in `src/viz.py:22-40` is for a conda-forge graphviz packaging bug. Add a `# TODO: remove once conda-forge graphviz ships a populated plugin config (tracked: <link to upstream issue>)` so a future maintainer knows to revisit.

**Deliverables.**

- Comment added to `src/viz.py`.

**Validation.**

- Reviewer reading the workaround knows when to remove it.

## Category E — Test coverage

Hygiene items to prevent silent regressions on the refactor work in Category A.

### E1. Tests for `_merged_evidence` ordering invariants

**Status.** ⬜ not started
**Resolves.** C11 from the review.

**Scope.** After A1 (split) lands and `_merged_evidence` lives in `app/state.py`, add a test suite covering soft↔hard transitions:

- Hard → soft → hard on the same node: final state is hard.
- Soft → hard → soft on the same node: final state is soft.
- Two soft observations on different nodes: both preserved.
- Removing an observation correctly recomputes the merged state.

**Deliverables.**

- `tests/test_state.py`.

**Validation.**

- Tests pass on the current `_merged_evidence` behaviour; protect against future regressions.

### E2. Tests for `src/viz.py` payload-building

**Status.** ⬜ not started
**Resolves.** C14 from the review.

**Scope.** Add unit tests for `build_agraph_payload` and `compute_node_levels` (post-A4):

- Returns the right number of nodes / edges.
- Every observed node has the observed-fill colour.
- Root drivers use the dedicated colour family.
- Topological levels are sane (Scenario at the highest level).

**Deliverables.**

- `tests/test_viz.py`.

**Validation.**

- Tests pass on the current implementation.

### E3. Tests for dashboard component helpers

**Status.** ⬜ not started
**Resolves.** C14 from the review.

**Scope.** After A1 (split) lands, add tests for:

- `_width_category` thresholds and boundary behaviour (post-C2 if the smooth gradient replaces buckets, test the gradient function instead).
- `_robustness_badge_html` rendering for narrow / moderate / fragile cases.
- `_ci_dataframe` row construction.
- Save / load session round-trip (`_save_named_session`, `_restore_named_session`).

**Deliverables.**

- `tests/test_components.py`.

**Validation.**

- Tests pass; coverage of `app/components/` rises above ~60%.

## Section B — Design decisions resolved

The decisions below are resolved.

1. **Refactor order — Decided: Category A before Category C.** Splitting the monolith and extracting the CSS first means Category C improvements drop into clean component modules rather than wading through the inline tangle.
2. **Engine caching pattern — Decided: pure-function queries.** Aligns with Plan 3's `Posterior` consumer pattern. Avoids per-session-state engine objects.
3. **Slider UX — Decided: drag-to-simplex + anchor mode, toggleable.** Both patterns offered; user picks per task.
4. **Robustness encoding — Decided: smooth gradient, retain emoji as coarse summary.** No hard category flips; emoji stays as a quick-glance signal.
5. **Colour palette — Decided: Wong's CVD-safe set.** `#0072B2`, `#E69F00`, `#D55E00`. Plus line-style encoding on charts.
6. **Edge rationale — Decided: hover tooltips in the DAG plus a tab for the full list.** Both surfaces; the tab handles omitted-edge discussion which is hard to do on hover.
7. **Scenario narratives — Decided: removed from always-on card, behind an expander, until roadmap B1 ships.** Compounds with B1 when it lands.
8. **`render_network_png` — Decided: keep with annotation pointing to roadmap B3.** Will become useful when session export ships.
9. **DAG canvas layout — Decided: defer flex/grid until UI framework migration.** Streamlit's primitives don't support it cleanly; document the brittleness and move on.
10. **Test coverage — Decided: add tests during/after each refactor item, not in a separate bulk pass.** Tests written when the helper is split out are higher quality than tests written months later.

## Section C — Open questions

| Question | Block | Notes |
| --- | --- | --- |
| Stacked-bar component implementation | C8 | Pure Altair? Custom Streamlit component? Decide before C8 starts. |
| CVD-safe palette validation | C9 | Need a stakeholder review with someone CVD-affected. Optional but worth it. |
| Flex/grid DAG canvas — defer or implement | C13 | Decided to defer; revisit if a UI-framework migration happens. |
| Bayes-factor display in observed-node panel | C4 | Depends on Plan 1 landing. If still in pgmpy-only mode, fall back to delta display. |

## Section D — Execution order summary table

| Order | Item | Category | Resolves | Rationale |
| --- | --- | --- | --- | --- |
| 1 | A1 — Split `dashboard.py` | Refactor | C2 | Foundation. Every subsequent visualization item benefits from clean module boundaries. |
| 2 | A2 — Extract CSS | Refactor | V8 | Trivial to do alongside A1; styles become independently editable. |
| 3 | A3 — Fix engine caching | Refactor | C1 | Prevents cross-session bleed in multi-user deployments. |
| 4 | A4 — Topological DAG levels | Refactor | C10 | Removes a manual-maintenance hazard. |
| 5 | B1 — Bound caches | Performance | C4 | One-line fix; prevents memory leaks in long sessions. |
| 6 | B2 — Memoise evolution series | Performance | C3 | Removes the largest interactive-latency source. |
| 7 | B3 — Apply-button override pattern | Performance | C3 partial, UX | Cleaner mental model and removes per-tick recomputation. |
| 8 | C1 — Right-size scenario cards | Visualization | V2 | Largest reallocation of viewport real estate. |
| 9 | C2 — Smooth robustness gradient | Visualization | V3 | Removes the hard-category-flip anti-pattern. |
| 10 | C3 — Drag-to-simplex sliders | Visualization | V4 | Largest single UX improvement to exploratory use. |
| 11 | C4 — Rich observed-node panel | Visualization | V5 | Most-interesting nodes get the most-informative panel. |
| 12 | C5 — DAG layout improvement | Visualization | V6 | Less clutter on the most-watched part of the graph. Automatic after A4. |
| 13 | C6 — Param vs forecast band distinction | Visualization | V7 | Removes a common stakeholder misreading. |
| 14 | C7 — Before/after delta on new obs | Visualization | V9 | Bridge to roadmap A3; immediately useful on its own. |
| 15 | C8 — Stacked-bar translator distributions | Visualization | V10 | Quicker scan in the Observations tab. |
| 16 | C9 — CVD-safe palette | Visualization | V11 | Accessibility; reach for ~5% of male stakeholders. |
| 17 | C10 — Multi-line tooltips | Visualization | V12 | Removes the headline-truncation cliff. |
| 18 | C11 — Edge rationale on hover | Visualization | V13 | Couples reasoning to the DAG it explains. |
| 19 | C12 — Responsive narratives | Visualization | V14 | Compounds with roadmap B1. Quick interim fix: hide them. |
| 20 | C13 — Flex/grid DAG canvas | Visualization | V15 | Likely deferred. |
| 21 | C14 — Continuous Oil_Price panel | Visualization | Plan 3 Phase 3/4 UI | Owns the continuous-viz work cleanly inside Plan 5 instead of inside Plan 3. Gated on Phase 3. |
| 22 | D1 — Dead code triage | Cleanup | C9 | Either delete or annotate `render_network_png`. |
| 23 | D2 — Drop `Tone` | Cleanup | C13 | Either thread or remove. |
| 24 | D3 — Document `+1e-6` guard | Cleanup | M8 | One-line comment. |
| 25 | D4 — Deduplicate sensitivity functions | Cleanup | M9 | Forwarder or deletion. |
| 26 | D5 — TODO on `dot -c` workaround | Cleanup | C12 | One-line comment. |
| 27 | E1 — `_merged_evidence` tests | Testing | C11 | Protects A3 refactor. |
| 28 | E2 — `viz.py` tests | Testing | C14 (review) | Protects A4 and C-category refactors. |
| 29 | E3 — Dashboard component tests | Testing | C14 (review) | Protects A1 and C-category refactors. |

---

**End of plan.** Companion plans: `docs/02_translator_robustification.md` (Plan 2, evidence ingestion), `docs/03_pymc_integration_plan.md` (Plan 3, inference engine), `docs/04_elicitation_tool_plan.md` (Plan 4, methodology layer). Orchestrator: `docs/master_plan.md`.
