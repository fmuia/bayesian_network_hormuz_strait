# Dashboard UI Plan — Deferred Remainder (commit-wise)

> **What this is.** The items of the dashboard plan **deferred out of the POC slice** ([`05_dashboard_ui_plan.md`](05_dashboard_ui_plan.md)) at the 2026-06-09 skeptical gate. Each is fully specified and written as a **commit-wise testable unit**, so it is ready to pick up when its trigger fires. Nothing here is abandoned — it is parked. Indexed from [`06_dropped_to_simplify.md`](06_dropped_to_simplify.md) §4.
>
> **Why these were deferred (the rule).** The POC ships only what a single-presenter committee demo makes a viewer *see* and *trust the model more for*. The items below are real and valuable but are one of: **(i) invisible in a demo** (architecture / performance / hygiene), **(ii) a secondary UX nicety** (the headline works without them), or **(iii) blocked by shelved upstream work** (Plan 3). Each card states which.
>
> **Status legend.** ⬜ not started · ◐ partially addressed · 🅿️ blocked (shelved upstream) · ✅ already addressed.
>
> **Commit convention.** As in the POC file: each card is an independently mergeable unit with an **acceptance gate** (test / `AppTest` / smoke check) and, where user-facing, a manual verification.

## Already addressed (no work needed)

- **B3 — apply-button override.** ✅ The override commits via a "Set observation" button (the sliders do not live-recompute the charts). The disabled-until-100 simplex friction was **also resolved** in improvements-1 (auto-normalise on apply — [`05_dashboard_ui_plan_improvements_1.md`](05_dashboard_ui_plan_improvements_1.md) I1); only the optional drag-on-triangle interaction remains parked in **R-C3**.
- **D3 — `+1e-6` Dirichlet guard.** ✅ near-done; already carries a `# avoid zero-alpha` comment in `src/sensitivity.py`. Only a one-line rationale expansion remains — **R-D3**.

## Group 1 — Architecture & performance *(invisible in a single-presenter demo)*

### ⬜ R-A3 — Fix the cached-but-mutated engine *(was A1/C1 · A3)*
**Why deferred.** The defect is **cross-session bleed**: the `@st.cache_resource` bootstrap engine is mutated (`clear_evidence` / `update_evidence`) on every rerun. This only manifests when **more than one user shares the process** (a hosted / multi-tenant deployment). A single-presenter demo never triggers it.
**Re-introduction trigger.** When the dashboard moves to a **hosted / multi-user** deployment.
**Scope.** A pgmpy-only pure-function `query(net, evidence, soft_evidence)` returning a posterior dict without mutating any cached object; **preserve Plan 4's locked-spec `get_engine` path** (a locked elicited network still wins over the bootstrap). Apply the same to the probability-evolution day loop. *(Plan 3's `Posterior` interface is shelved, so this is pgmpy-only — no backend abstraction.)*
**Acceptance gate.** A test runs two independent evidence states through `query(...)` and asserts neither leaks into the other; the canonical-state dashboard outputs are unchanged.

### ⬜ R-A4 — Topological DAG levels *(C10 · A4)*
**Why deferred.** A **maintenance hazard, not a visible defect** — the hardcoded levels render correctly today; the only risk is that a *future* added node silently lands at level 0.
**Re-introduction trigger.** Next time a node is added to the network, or a second network is onboarded.
**Scope.** `compute_node_levels(edges) -> dict[str, int]` (longest path from any root) replacing **both** `_NODE_LEVEL` and `_NODE_LEVEL_LATENT`; `TOPOLOGY_LAYOUT` derives levels rather than hardcoding them. Memoise per topology; fallback to level 0 + a logged warning for unplaced nodes.
**Acceptance gate.** Reproduces both hardcoded dicts exactly, **per topology** (note they differ: `Scenario` is level 4 in labelling, level 3 in latent); a synthetic added node gets a sane level.

### ⬜ R-B1 — Bound the Streamlit caches *(C4-finding · B1)*
**Why deferred.** A **long-session memory** concern; a demo session is short. Cheap to add later.
**Re-introduction trigger.** A long-running / always-on deployment, or observed memory growth.
**Scope.** `max_entries=64, ttl="1h"` on `cached_credible_intervals` and `cached_node_credible_intervals` (the two `@st.cache_data` helpers, which key on evidence + `topology` + `locked_spec_json`).
**Acceptance gate.** Bounded memory across a long slider-tweak loop (process-monitor smoke check).

### ⬜ R-B2 — Memoise the evolution series on observation IDs *(C3-finding · B2)*
**Why deferred.** A **latency** optimisation, only worth it if the per-day recompute feels slow in the demo. **If it does, promote this into the POC** — it is the cheapest single latency win.
**Re-introduction trigger.** The evolution chart visibly lags on commit, or sessions grow long.
**Scope.** `@st.cache_data` keyed on `tuple(sorted(obs["id"] for obs in observations))`, returning the long-form DataFrame; new observations invalidate naturally, slider tweaks do not.
**Acceptance gate.** Cache-hit logging shows hits on slider tweaks and misses only on new observations; visual parity with the current chart.

## Group 2 — Secondary UX *(the headline works without these)*

### ◐ R-C3 — Drag-to-simplex override sliders *(V4 · C3)* — friction half done
**Why deferred.** The override is a **power-user feature**, not the demo headline; the sliders already work and already button-commit (B3 done).
**Partially addressed (improvements-1, 2026-06-11).** The disabled-until-100 simplex friction — the actual **V4** trap — is gone: the override now **auto-normalises on apply** (`state.override_to_observation`), so any positive slider mix commits as a normalised soft distribution (or a hard pin for a single non-zero state). See [`05_dashboard_ui_plan_improvements_1.md`](05_dashboard_ui_plan_improvements_1.md) I1.
**Re-introduction trigger.** Analysts use the override heavily and want a faster *interaction* than per-state sliders.
**Scope (remaining).** Only the optional drag-on-triangle + anchor-distribute interaction; the disabled-until-100 anti-pattern is already removed.
**Acceptance gate.** A helper test: dragging one state rescales the others; soft evidence commits correctly to session state.

### ⬜ R-C6 — Param-vs-forecast band styling *(V7 · C6)*
**Why deferred.** **Subtle**; the caption already names the band, so the misread risk is partly mitigated. Low demo impact.
**Re-introduction trigger.** Stakeholders misread the band despite the caption, or a true forecast band ships.
**Scope.** Hashed / dotted band fill + an anchored "Band = robustness; not a forecast" annotation in `evolution_chart.py`.
**Acceptance gate.** Visual: a test stakeholder reads the band as non-forecast without the caption.

### ⬜ R-C8 — Stacked-bar `state_probs` *(V10 · C8)*
**Why deferred.** **Minor** — assignments are already styled chips; only the per-assignment *distribution* is text. Nice, not essential.
**Re-introduction trigger.** Analysts scan the translation panel often and want the distribution at a glance.
**Scope.** Inline ~60px stacked bar per `state_probs`; fold the structured-pipeline (claims→mappings) detail into the same `translator_panel.py` so the per-observation view is one coherent panel. Text retained on hover.
**Acceptance gate.** A helper test builds the stacked-bar data correctly; the text values remain accessible on hover.

### ⬜ R-C9 — Colourblind-safe palette *(V11 · C9)* — moved out of the POC 2026-06-10
**Why deferred.** **Forgone for the first pass** (analyst decision): the current green/amber/red reads fine for a single-presenter demo; CVD-safe colours are an accessibility refinement, not a comprehension blocker. The P6 smooth-robustness gradient already removed the hard-bucket flip on its own anchors.
**Re-introduction trigger.** Accessibility becomes a requirement (a CVD-affected stakeholder, a public/contractual deployment, or a greyscale-print need).
**Scope.** Centralise `SCENARIO_COLOR` (in `app/theme.py`) to Wong's set `#0072B2` / `#E69F00` / `#D55E00`; update every Altair scale; add solid/dashed/dotted line styles on the evolution chart; re-anchor `ci_charts.robustness_color`; sweep the Triage / relevance / stream surfaces (the C15 colour half).
**Acceptance gate.** A test asserts the palette constants + that every scenario chart uses the scale; a CVD-simulator (Coblis) check shows the three scenarios stay distinct in greyscale.

### ⬜ R-C10 — Multi-line evolution tooltips *(V12 · C10)*
**Why deferred.** **Minor** truncation polish.
**Re-introduction trigger.** Tooltips clip headlines in real use.
**Scope.** Remove the 180-char cap; one headline per line in the Altair tooltip.
**Acceptance gate.** A day with three headlines shows all three in full in the tooltip.

### ⬜ R-C13 — Flex/grid DAG canvas *(V15 · C13)*
**Why deferred.** **Already decided to defer** — Streamlit's layout primitives don't support it cleanly; the brittle 460px height is tolerable until a UI-framework migration.
**Re-introduction trigger.** A move off Streamlit, or the layout breaks on a target viewport.
**Scope.** A flex/grid canvas height that adapts to the right column, **or** a documented TODO at the hardcoded value.
**Acceptance gate.** If built: the canvas matches the right-column height on common viewports. If deferred-in-place: the TODO is present.

### 🅿️ R-C14 — Continuous Oil_Price panel and interval queries *(C14)*
**Why deferred.** **Blocked** — depends on Plan 3 Phase 3/4 (continuous nodes in a PyMC backend), which is **shelved** (see [`06_dropped_to_simplify.md`](06_dropped_to_simplify.md) §2). The discrete bar-chart fallback (current behaviour) stays.
**Re-introduction trigger.** Plan 3 is revived and promotes `Oil_Price` to a continuous variable.
**Scope.** `app/components/continuous_viz.py` — density plot + interval-query widget, behind a `Posterior` capability flag; interval-probability readouts on the cards.
**Acceptance gate.** Discrete backend: existing bar-chart behaviour preserved. Continuous backend: density plot renders; interval queries match direct `probability_of_interval` calls.

## Group 3 — Code hygiene *(invisible)*

### ⬜ R-D1 — `render_network_png` dead code *(C9-finding · D1)*
**Why deferred.** **Invisible cleanup.** Imported by `app/dashboard.py` but unused (the dashboard renders via `streamlit-agraph`); harmless.
**Re-introduction trigger.** Opportunistic, or when roadmap B3 (session export to PDF/HTML) needs it.
**Scope.** Keep with a `# Used by: bn_app_next_steps.md B3 — session export` annotation, or delete it and drop the import.
**Acceptance gate.** Either the annotation lands, or deletion leaves the suite green.

### ⬜ R-D2 — Drop `Observation.tone` *(C13-finding · D2)*
**Why deferred.** **Type-vocabulary noise**, invisible. `Tone` + `Observation.tone` are used only by `ExampleHeadline`.
**Re-introduction trigger.** Opportunistic, or when roadmap B1 (daily narratives) threads tone.
**Scope.** Remove `Tone` + the `Observation.tone` field, or thread it into narrative generation.
**Acceptance gate.** Suite green; no callers broken.

### ⬜ R-D3 — Expand the `+1e-6` guard comment *(M8 · D3 residual)*
**Why deferred.** ✅ Largely done — the guard already carries a `# avoid zero-alpha` comment. Only a one-line rationale expansion remains.
**Re-introduction trigger.** Opportunistic.
**Scope.** Expand the comment in `src/sensitivity.py` to name the future-CPT-structural-zero intent.
**Acceptance gate.** Suite green.

### ⬜ R-D4 — Deduplicate the sensitivity functions *(M9 · D4)*
**Why deferred.** **Maintenance**, invisible — two correct implementations of the same Monte-Carlo loop to keep in sync.
**Re-introduction trigger.** Either function changes, or a third caller appears.
**Scope.** `scenario_credible_intervals` forwards to `node_credible_intervals(..., nodes=["Scenario"])` (or is deleted, callers updated).
**Acceptance gate.** `tests/test_sensitivity.py::test_node_ci_matches_scenario_ci_for_scenario_node` stays green.

### ⬜ R-D5 — TODO on the `dot -c` workaround *(C12-finding · D5)*
**Why deferred.** A **one-line** reminder, invisible.
**Re-introduction trigger.** Opportunistic.
**Scope.** Add a `# TODO: remove once conda-forge graphviz ships a populated plugin config (tracked: <upstream issue>)` to the `_PLUGINS_REGISTERED` / `dot -c` workaround in `src/viz.py`.
**Acceptance gate.** Comment present.

## Deferred execution order (when picked up)

| Order | Commit | Trigger class |
|---|---|---|
| 1 | R-B2 — memoise evolution series | promote early if the demo lags |
| 2 | R-A3 — engine caching | hosted / multi-user |
| 3 | R-A4 — topological levels | next node added |
| 4 | R-B1 — bound caches | long-running deployment |
| 5 | R-C3 — drag-to-simplex *(interaction only; sum-to-100 friction done)* | heavy override use |
| 6 | R-C6 — band styling | observed misreads |
| 7 | R-C8 — stacked-bar `state_probs` | opportunistic |
| 7b | R-C9 — CVD-safe palette | accessibility requirement |
| 8 | R-C10 — multi-line tooltips | observed clipping |
| 9 | R-D1…R-D5 — hygiene | opportunistic |
| 10 | R-C13 — flex/grid canvas | UI-framework migration |
| 11 | 🅿️ R-C14 — continuous Oil_Price | Plan 3 revived |

## Provenance

The full original diagnosis (26 findings), the ten resolved design decisions, and the open-questions table are preserved in the git history of `05_dashboard_ui_plan.md` prior to the 2026-06-09 split, and the canonical finding registry is `master_plan.md` §4. The POC slice closes V2/V3/V5/V8/V9/V11/V13/V14 + the architecture findings; this file carries the remainder.
