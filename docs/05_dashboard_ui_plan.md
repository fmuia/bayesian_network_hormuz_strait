# Dashboard UI Plan — POC Slice (commit-wise)

> **What this is.** The **stakeholder-POC execution slice** of the dashboard plan, reordered into commit-by-commit **testable units**. The deferred remainder (with per-item reasons + triggers) lives in the companion file [`05_dashboard_ui_plan_deferred.md`](05_dashboard_ui_plan_deferred.md). Reconciled against the merged code on `explorations-dev-plan-5`, **2026-06-09**.
>
> **Why a POC slice.** After a skeptical gate (the same one applied to Plan 2), Plan 5's 29 items split into the demo-critical subset shipped here and a deferred backlog. An item earns POC inclusion only if a committee viewer **(a) sees it** and **(b) trusts/understands the model more for it** — or it is **foundation the visible items can't be built cleanly without**. Everything else is deferred *with a trigger* in the companion file; nothing is deleted.
>
> **Scope guards.** A single-presenter committee demo, on the existing **pgmpy** path and the **latent-regime** default topology. No PyMC (Plan 3 shelved). No multi-user / hosted concerns (that is the deferred A3).
>
> **Status legend.** ⬜ not started · ⏳ in progress · ✅ shipped (with date).
>
> **Commit convention (testable units).** Each `P*` commit below is an **independently mergeable, runnable vertical slice** with (i) an **automated acceptance gate** (unit tests / `AppTest`) and (ii) a **manual verification** step (`pixi run app`). The app renders at every boundary. Tests are written *with* the refactor that introduces the helper, not in a later bulk pass.

## What the POC closes

| Finding | Item | One-liner |
|---|---|---|
| C2 | A1 | split the 2,350-line monolith into a component library |
| V8 | A2 | extract the ~356 lines of inline CSS |
| (new) | A5 | consolidate the five-view IA; fix the 🧪 icon collision |
| V2 | C1 | right-size scenario cards, enlarge the evolution chart |
| V3 | C2 | smooth robustness gradient (no hard emoji flips) |
| V5 | C4 | rich observed-node panel + first-class Bayes-factor contribution (Plan 1) |
| V9 | C7 | before/after delta chips on a new observation |
| V13 | C11 | edge rationale on hover in the DAG |
| V14 | C12 | hide the static scenario narratives (interim) |
| (new) | C15 | consistency pass so the Plan-2/4 surfaces match |
| C11/C14 (review) | E1–E3 | tests for state, viz, and component helpers (written alongside the split) |

(B3 was already satisfied and D3 is near-done — see the deferred file's "already addressed" note. The full 26-finding diagnosis is preserved in git history and in `master_plan.md` §4.)

## Design decisions in force (POC subset)

1. **Category-A refactor precedes the visuals** — split + CSS first, so each visual lands in a clean module.
2. **Engine caching** — *deferred* (A3, R-A3 in the companion): the POC keeps the current cached-engine pattern, which is safe for a single presenter.
3. **Robustness encoding** — smooth gradient; emoji retained as a coarse summary.
4. **Palette** — *deferred for the first pass* (R-C9 in the companion): keep the current green/amber/red; the Wong CVD-safe set + line-style encoding lands when accessibility becomes a requirement.
5. **Scenario narratives** — removed from the always-on card, behind an expander.
6. **Edge rationale** — hover tooltips in the DAG **plus** the existing tab (the tab keeps the omitted-edges discussion).
7. **Bayes-factor panel** — first-class (Plan 1 latent default; `src/inference.py:scenario_bayes_factors`), guarded to the latent topology (it raises on labelling).
8. **Tests written per-commit**, leveraging the existing `AppTest` harness (`tests/test_hitl.py`, `tests/elicitation/test_dashboard.py`).

## Commit-wise execution plan

### Phase 1 — Foundation (split, CSS, IA)

#### ⬜ P1 — Extract CSS to `app/styles.css` (A2 · V8)
**Scope.** Move the single inline `st.markdown("<style>…")` block (~356 lines) to `app/styles.css`; load it at startup. Organise with section comments matching the current structure (sidebar, cards, sliders, translator stream, and the `.assign-chip` / Triage / structured / eval-badge classes).
**Acceptance gate.** `AppTest` boots exception-free and the page renders; no inline `<style>` block remains in `app/dashboard.py`; `app/styles.css` is loaded.
**Manual verification.** `pixi run app` — visual parity; editing `styles.css` + reload changes styling without a Python edit.

#### ⬜ P2 — Extract `app/state.py` + state tests (A1a · E1 · E3-state)
**Scope.** Move session defaults (incl. `review_queue`, `locked_spec_json`), `_merged_evidence`, `_append_observation`, named-session save/load, and the Triage helpers (`_build_review_item`, `_inject_review_item`, `_remove_from_review`) into `app/state.py`. Keep them pure (no Streamlit context where avoidable).
**Acceptance gate.** New `tests/test_state.py` (importable without a Streamlit context) covers the `_merged_evidence` soft↔hard ordering invariants (hard→soft→hard, soft→hard→soft, two-node, removal-recompute), the save/load round-trip, and the Triage helpers; full suite green.
**Manual verification.** Translate / override / save / load / Triage all behave as before.

#### ⬜ P3 — Extract chart components + viz tests (A1b · E2 · E3-charts)
**Scope.** Extract `app/components/scenario_cards.py`, `ci_charts.py` (dumbbell + robustness badge + `_ci_dataframe` + `_width_category`), and `evolution_chart.py`. Add `tests/test_viz.py` for `build_agraph_payload` (node/edge counts, observed-fill colour, root-driver colour family) on **both** topologies.
**Acceptance gate.** `tests/test_viz.py` + chart-helper tests green; `AppTest` renders the pinned band identically.
**Manual verification.** Cards + CI dumbbell + evolution chart unchanged.

#### ⬜ P4 — Extract the remaining views; reduce `dashboard.py` to orchestration (A1c · C2)
**Scope.** Extract `network_view.py`, `observation_log.py`, `edge_rationale.py`, `audit_view.py`, `triage_view.py`, `translator_stream.py`, `structured_panel.py`. `app/dashboard.py` keeps page setup, session wiring, and the `st.segmented_control` routing only. `app/elicitation_panel.py` (Plan 4) stays; align its imports/styles.
**Acceptance gate.** `AppTest` renders every nav view exception-free; `app/dashboard.py` drops below ~300 lines; each component's pure helpers import without a Streamlit context.
**Manual verification.** Every view + the elicitation expander render and behave as before.

#### ⬜ P5 — Consolidate the information architecture (A5 · new)
**Scope.** Fix the **🧪 emoji collision** (the Triage view and the Elicitation expander both use 🧪 → distinct icons); ensure every top-level surface registers through the one `st.segmented_control`; record the IA decision (which surface is nav vs sidebar vs expander) in an `app/components/README`.
**Acceptance gate.** `AppTest` asserts the nav lists all views with **distinct** icons; no two surfaces share an icon.
**Manual verification.** Nav reads cleanly; each surface has one obvious home.

### Phase 2 — Demo visuals (each a testable commit on the now-clean components)

#### ⬜ P6 — Smooth robustness gradient (C2 · V3)
**Scope.** Replace the 3-bucket 🟢🟡🔴 (`±8/±20`) with a continuous LERP over half-width in `ci_charts.py`; keep a coarse emoji + a numeric readout.
**Acceptance gate.** Unit test of the gradient function: monotonic colour vs half-width; boundary cases (7pp vs 9pp produce *close* colours, not a category flip).
**Manual verification.** A node moving 7→9pp shows a smooth shift, not a flip.

#### ⬜ P7 — Right-size cards, enlarge the evolution chart (C1 · V2)
**Scope.** Compress the scenario cards to a vertical strip (~1/5 viewport) in `scenario_cards.py`; reallocate space to the evolution chart.
**Acceptance gate.** `AppTest` renders the new layout; card-helper tests still pass.
**Manual verification.** Evolution chart ≥1.5× larger; cards legible at common breakpoints.

#### ⬜ P8 — Rich observed-node panel + Bayes-factor (C4 · V5)
**Scope.** Replace the flat 100%/0% bar with `app/components/observed_node_panel.py`: observed value + source + day; downstream posterior change; previous posterior; and the **Bayes-factor contribution** (`scenario_bayes_factors`, guarded to the latent topology).
**Acceptance gate.** Component test: the panel reads value/source/day; the Bayes-factor wiring returns values on the latent topology and is cleanly skipped on labelling.
**Manual verification.** On the canonical escalation, the observed-node panel shows the contribution.

#### ⬜ P9 — Before/after delta chips (C7 · V9)
**Scope.** Store previous-observation scenario probabilities in `state.py`; render `▲ +5pp Severe` / `▼ -3pp Stress` chips on the cards when a new observation commits.
**Acceptance gate.** Unit test of the delta computation (prev vs current → signed pp); undo/remove restores the prior state cleanly.
**Manual verification.** Committing an observation shows visible deltas; removing it clears them.

#### ⬜ P10 — Edge rationale on hover (C11 · V13)
**Scope.** Populate `streamlit_agraph` edge titles from `_EDGE_RATIONALE` in `build_agraph_payload`; keep the Edge-rationale view for the full list + omitted edges.
**Acceptance gate.** `tests/test_viz.py` asserts every edge in the payload carries its rationale title.
**Manual verification.** Hovering any edge shows the rationale.

#### ⬜ P11 — Hide the static narratives (C12 · V14)
**Scope.** Move the fixed scenario-narrative paragraph off the always-on card into an expander.
**Acceptance gate.** `AppTest`: the card no longer always-renders the narrative; it is reachable via the expander.
**Manual verification.** Cards denser; narrative on click.

#### ⬜ P12 — Consistency pass over the Plan-2/4 surfaces (C15 · new)
**Scope.** Apply the current palette, the smooth robustness encoding (P6), and card density to the **Triage**, translator-relevance, structured-pipeline, and **elicitation** surfaces; ensure they consume `styles.css` (P1) and the component conventions (P4). *(The CVD-safe palette sweep is deferred with R-C9.)*
**Acceptance gate.** `AppTest` renders all surfaces; a visual pass confirms one palette + one robustness encoding across surfaces.
**Manual verification.** Triage / translator / structured / elicitation match the scenario cards' look.

## Execution order summary

| # | Commit | Maps to | Resolves |
|---|--------|---------|----------|
| P1 | Extract CSS | A2 | V8 |
| P2 | `app/state.py` + state tests | A1, E1, E3 | C2, C11 |
| P3 | Chart components + viz tests | A1, E2, E3 | C2, C14(review) |
| P4 | Remaining views; orchestration-only `dashboard.py` | A1 | C2 |
| P5 | IA consolidation (🧪 fix) | A5 | (new) |
| P6 | Smooth robustness gradient | C2 | V3 |
| P7 | Right-size cards / bigger evolution chart | C1 | V2 |
| P8 | Rich observed-node panel + Bayes | C4 | V5 |
| P9 | Before/after delta chips | C7 | V9 |
| P10 | Edge rationale on hover | C11 | V13 |
| P11 | Hide static narratives | C12 | V14 |
| P12 | Consistency pass | C15 | (new) |
| ~~CVD-safe palette~~ | 🅿️ **deferred 2026-06-10** (R-C9) | C9 | V11 |

## After the POC

The deferred remainder — engine-caching for multi-user (A3), topological DAG levels (A4), cache bounds + evolution memoisation (B1/B2), drag-to-simplex sliders (C3), param/forecast band styling (C6), stacked-bar `state_probs` (C8), **CVD-safe palette (C9, deferred 2026-06-10)**, multi-line tooltips (C10), flex/grid canvas (C13), continuous Oil_Price (C14, Plan 3), and the D-hygiene items — lives in [`05_dashboard_ui_plan_deferred.md`](05_dashboard_ui_plan_deferred.md), each with its deferral reason, re-introduction trigger, and a commit-wise acceptance gate. Indexed from [`06_dropped_to_simplify.md`](06_dropped_to_simplify.md) §4.
