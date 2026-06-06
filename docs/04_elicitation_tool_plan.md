# Elicitation Tool Plan: A Multi-Protocol CPT Elicitation Platform

> **Status.** Draft. No layers started.
>
> **Position in the sequencing.** Third of three sequential plans. Depends on `docs/02_translator_robustification.md` (Plan 2) for the audit-log substrate (item D3) and on `docs/03_pymc_integration_plan.md` (Plan 3) for the `NetworkSpec` declarative model representation. Run after both are complete.
>
> **Related docs.** `docs/master_plan.md` §4 is the in-tree registry of finding IDs (this plan closes the elicitation-related M-findings, notably M3 and M6, plus C2 from the roadmap). `docs/02_translator_robustification.md` builds the audit log, versioned-artefact pattern, HITL review queue, and source-credibility infrastructure that this plan reuses. `docs/03_pymc_integration_plan.md` provides the `NetworkSpec` interface that elicited CPTs are exported to. `docs/01_latent_regime_plan.md` Section A.8 covers the math context for why per-CPT $\kappa$ values and hierarchical priors matter.
>
> **Status legend.** ⬜ not started · ⏳ in progress · ✅ shipped (with date).

## Executive Summary

This plan delivers the **elicitation methodology layer** for the platform: a multi-protocol tool for eliciting, aggregating, versioning, and calibrating the conditional probability tables that drive the Bayesian network. It addresses the deepest epistemic weakness of the current model — that CPT values are inline literals chosen by one author without formal protocol, multi-expert input, or calibration tracking.

The plan is structured as **six layers in scope** (Layers 0 through 5) plus a deferred commercial Layer 6 (billing, onboarding, support) that ships only when paying customers exist. Layer 0 is the data model and storage substrate that extends Plan 2's audit log; Layer 1 is the mathematical engine (aggregation primitives, ranked nodes, sensitivity analysis); Layer 2 implements three elicitation protocols (Cooke's classical model, IDEA, SHELF) as configurable workflows; Layer 3 is the Streamlit UI; Layer 4 integrates with Plan 3's inference engine; Layer 5 adds advanced features (LLM-proposed CPTs, calibration tracking, ranked-node UI).

An optional extension, **Layer 5.5 (agentic AI experts)**, lets LLM agents *participate in the protocols as contributors* — but only as **calibration-validated** ones: scored on the same seed questions as human experts (reusing the Cooke machinery), weighted by measured performance, composed across diverse base models to control correlated error, and always attributable in provenance with human sign-off for high-stakes work. The positioning is **calibration-validated AI elicitation**, not "AI replaces experts." The peer-reviewed methodology this rests on — structured expert judgment, LLM forecasting/calibration, multi-agent aggregation, and LLM-elicited priors — is catalogued with primary sources in **Section E**, each annotated with how it supports the design and where further research or stronger validation is warranted.

The platform is positioned as **methodology-as-product** with an **open-core licensing model**: the inference engine, mathematical primitives, and protocol implementations are open-source; the commercial layer (deployment automation, support, hosted version, premium integrations) is closed. Deployment shape is **multi-deployment, single-tenant per deployment** — each customer engagement gets its own isolated stack rather than sharing infrastructure SaaS-style. This matches the high-stakes regulatory and consulting-led nature of the use cases.

## Context

### Position in the broader plan stack

Three plans run sequentially:

1. **Plan 2 — `docs/02_translator_robustification.md`**: fixes the evidence ingestion layer. Establishes the audit-log substrate, versioned-artefact pattern, HITL review queue, source-credibility infrastructure, and golden-set evaluation harness.
2. **Plan 3 — `docs/03_pymc_integration_plan.md`**: rebuilds the inference layer. Introduces the `NetworkSpec` declarative model, dual-backend dispatch (pgmpy / PyMC), hierarchical priors over CPTs, support for continuous variables, and the latent-regime topology.
3. **Plan 4 (this doc)**: builds the elicitation methodology layer on top of the substrate from Plan 2 and the model spec from Plan 3.

Each plan is fully self-contained internally; the three together cover evidence ingestion, inference, and methodology — the three layers of a defensible scenario-modelling platform.

### Position in the market

Examined against existing tools:

- **AgenaRisk, Netica, Hugin, GeNIe.** General-purpose Bayesian-network tools with CPT-editor UIs and (in AgenaRisk's case) ranked-node compression. Strong on the mechanical CPT-editing surface; weak on multi-protocol elicitation methodology, multi-expert aggregation, and calibration tracking. We borrow the ranked-node methodology and the visual CPT-editor pattern; we replace the spreadsheet-style elicitation UI with protocol-driven workflows.
- **Good Judgment Inc., Hypermind, Metaculus.** Forecaster ensembles with calibration tracking. Excellent on the calibration side; not designed for structured CPT elicitation in BN contexts. We borrow the calibration-tracking pattern (Cooke weights from realised performance) and apply it to per-expert CPT contributions.
- **Catastrophe modelling (RMS, AIR, CoreLogic).** Heavy expert elicitation for tail risk where no calibration data exists. Methodology mature; tooling proprietary. We borrow the anchored-elicitation-against-analogs pattern and the sensitivity-driven prioritization approach.
- **Pharmaceutical / regulatory (SHELF, EFSA, IPCC).** Open methodology documents on how to elicit defensibly. We implement the protocols directly (SHELF, IDEA, Cooke).
- **Geopolitical intel platforms (Recorded Future, RANE, Maplecroft).** Productised but methodologically opaque. Our positioning is the inverse: methodologically transparent, productised for high-stakes consulting use.

The market gap we occupy: **methodologically rigorous, multi-protocol, calibration-aware CPT elicitation, productised for high-stakes consulting and regulatory contexts.** No off-the-shelf tool covers this space.

## Diagnosis: Why the Current State Is Insufficient

The list below is the failure surface this plan closes. Items marked (M*) are finding IDs from the master-plan §4 matrix.

1. **CPT values are inline literals chosen by one author without protocol.** `src/network.py` contains hand-tuned probability values with brief Python comments as justification. No record of who picked the numbers, when, against what reference, or with what confidence. The README acknowledges this explicitly.
2. **No multi-expert aggregation.** Even when multiple analysts have views on a CPT entry, there is no infrastructure to elicit them independently and aggregate. A single author's blind spots become the model's blind spots.
3. **No calibration tracking.** The model has been running on demo evidence for months. There is no record of which CPT entries produced predictions that matched outcomes, or which produced predictions that failed.
4. **No CPT versioning.** When the author tweaks a value, the previous value is lost to git history at best. Stakeholders cannot ask "what did we believe about this last quarter?" or compare model output across two parameter sets.
5. **No provenance per CPT entry.** Every cell in the 13 CPTs has the same epistemic status from the consumer's perspective: "the author chose this." A defensible model needs per-cell provenance: who elicited, when, with what method, against what reference, with what confidence.
6. **No sensitivity-driven prioritization.** Effort to improve CPTs is distributed uniformly, but in any BN a small subset of CPT entries drives most of the output variation. Without sensitivity analysis, elicitation effort goes to the wrong places.
7. **No support for ranked-node compression.** A 27-column CPT is brutal to elicit cell-by-cell. The Fenton & Neil ranked-node methodology reduces this to a handful of weights, but it's not implemented anywhere.
8. **No support for multiple protocols.** Different stakeholders demand different rigour. Regulators expect Cooke's classical model; corporate boards accept IDEA; a single analyst can do SHELF. The current implementation supports none of them explicitly.
9. **No reuse across customer engagements.** Each new client problem would today require copy-paste-and-edit of `network.py`. A productised platform requires per-deployment isolation and per-engagement configuration.
10. **No coupling to the LLM translator's evidence corpus.** Plan 2 builds an audit log of every article translated, with span-grounded claims and analyst-approved corrections. This corpus is exactly the analog-event database that anchored elicitation needs, but it's not currently usable for CPT elicitation.

## Architecture: Layered Platform Design

### Design principles

1. **Methodology-as-product, open-core.** The inference engine, mathematical primitives, protocol implementations, and aggregation logic are open-source under a permissive license. The commercial layer — deployment automation, hosted versions, premium integrations, support — is closed. This is the standard pattern for PPL-adjacent B2B tooling.
2. **Multi-deployment, single-tenant per deployment.** Each customer engagement gets its own isolated stack: own database, own app instance, own users, own configuration. The code is shared across engagements; the deployments are not. No tenant_id columns; no row-level security; no SaaS-style data co-mingling. This matches the high-stakes regulatory and consulting-led nature of the use cases.
3. **Layered architecture with foundation-first ordering.** Each layer is independently testable and deliverable. Lower layers do not depend on higher ones; higher layers consume lower ones via stable interfaces.
4. **Shared infrastructure with Plan 2.** The audit log schema, versioned-artefact pattern, HITL review queue, source-credibility registry, and Streamlit UI shell are extended rather than duplicated.
5. **Shared interface with Plan 3.** Elicited CPTs export to the `NetworkSpec` from `docs/03_pymc_integration_plan.md`. Per-CPT $\kappa$ values respected by both backends.
6. **Streamlit-first UI.** Streamlit forms for v1; upgrade to a dedicated web frontend when scaling or feature demands require it. Do not over-engineer the UI before product-market fit.

### The layered structure

```
┌──────────────────────────────────────────────────────────────────┐
│ Layer 6 — Commercial layer (deferred)                            │
│  Billing · onboarding · documentation · tenant config UI          │
├──────────────────────────────────────────────────────────────────┤
│ Layer 5 — Advanced features                                       │
│  LLM-proposed CPTs · ranked-node UI · sensitivity-driven priority │
│  · calibration tiers 2/3 · Cooke weight updates from outcomes     │
├──────────────────────────────────────────────────────────────────┤
│ Layer 4 — Integration with inference                              │
│  Elicited CPTs → NetworkSpec → PgmpyBackend / PymcBackend         │
├──────────────────────────────────────────────────────────────────┤
│ Layer 3 — UI (Streamlit)                                          │
│  Protocol workflows · CPT review · Sources · HITL · calibration   │
├──────────────────────────────────────────────────────────────────┤
│ Layer 2 — Protocol implementations                                │
│  Cooke · IDEA · SHELF as configurable workflows                   │
├──────────────────────────────────────────────────────────────────┤
│ Layer 1 — Core engine                                             │
│  Aggregation primitives · ranked nodes · sensitivity analysis     │
├──────────────────────────────────────────────────────────────────┤
│ Layer 0 — Data model and storage substrate                        │
│  Multi-deployment isolation · schema · auth · audit log extension │
└──────────────────────────────────────────────────────────────────┘
```

### Data model: multi-deployment single-tenant

Each customer engagement runs an isolated stack. Within a single deployment, the schema is:

| Table | Purpose |
| --- | --- |
| `users`, `roles`, `permissions` | RBAC primitives for this deployment |
| `networks` | This deployment's DAGs (one or more; e.g., a Hormuz network plus variants) |
| `cpts` | CPT current values, indexed by network + node |
| `cpt_versions` | Historical CPT values with full audit trail |
| `cpt_provenance` | Per-CPT-version metadata: protocol used, elicitor(s), date, references |
| `experts` | Registered domain experts for this deployment, with calibration history |
| `expert_calibration` | Per-expert performance on calibration questions (Cooke weights) |
| `elicitation_sessions` | Protocol runs (Cooke / IDEA / SHELF) with all inputs and aggregated outputs |
| `elicitation_session_events` | State machine events for resumable workflows |
| `articles` | Translator audit log (shared schema with Plan 2) |
| `translations` | Per-article translation outputs (Plan 2's D3) |
| `analyst_actions` | HITL review log (shared with Plan 2) |
| `sources` | Per-source credibility registry (shared with Plan 2) |
| `outcomes` | Realised intermediate-node states for Tier 2 calibration |
| `calibration_runs` | Scheduled evaluation results |

The tables marked "shared with Plan 2" already exist after Plan 2 is complete; this plan extends them with elicitation-specific columns or adjacent join tables rather than duplicating.

No `tenant_id` column anywhere. Each deployment's database is one customer's data.

Storage: SQLite for development; Postgres for production deployments. ORM: SQLAlchemy or SQLModel. Migrations: Alembic.

## Section A — Layered plan

Each layer has a clear scope, deliverables, and validation criteria. Layers 0–3 deliver a minimum viable elicitation platform (the v1 milestone). Layers 4–5 add structural and intelligent features. Layer 6 is deferred.

### Layer 0 — Data model and storage substrate

**Status.** ⬜ not started
**Resolves.** Diagnosis items 4 (no CPT versioning), 5 (no provenance), 9 (no reuse across engagements). Establishes the platform foundation.

**Scope.** Define and ship the elicitation-tool schema as an extension of Plan 2's audit log schema. Per-deployment isolated database. Auth scaffolding (per-deployment user table, with hooks for SSO integration). Configuration system for per-deployment customisation (network choice, protocol availability, branding).

The schema additions in this layer:

- `experts` and `expert_calibration` (Cooke weights track record).
- `elicitation_sessions` and `elicitation_session_events` (resumable protocol runs).
- `cpts`, `cpt_versions`, `cpt_provenance` (versioned CPT history with full audit trail).
- `outcomes` (post-hoc-labelled outcomes for calibration).
- `calibration_runs` (scheduled evaluation results).

Plus Alembic migrations and ORM models for the above.

**Deliverables.**

- `src/elicitation/db/schema.py` — SQLAlchemy / SQLModel definitions.
- `src/elicitation/db/migrations/` — Alembic migrations.
- `src/elicitation/auth/` — per-deployment user/role tables and login plumbing.
- `src/elicitation/config/` — per-deployment YAML configuration loader.
- Documentation: `docs/deployment.md` — how to stand up a new customer deployment.

**Validation.**

- Schema migrates cleanly on a fresh SQLite database and a fresh Postgres database.
- Two parallel deployments (e.g., `client_a.db` and `client_b.db`) operate without any cross-database visibility.
- Migration round-trip: applying then rolling back leaves the database in its pre-migration state.
- Documentation walks through a new-customer deployment end-to-end.

### Layer 1 — Core engine: aggregation primitives, ranked nodes, sensitivity

**Status.** ⬜ not started
**Resolves.** Diagnosis items 6 (no sensitivity prioritization), 7 (no ranked-node compression). Provides the mathematical primitives that protocols consume.

**Scope.** Pure-function library with no database or UI dependencies. Three subsystems:

1. **Aggregation primitives.** Linear pooling, geometric (logarithmic) pooling, performance-weighted aggregation (Cooke). Each as a pure function over expert distributions.
2. **Ranked-node implementation.** Fenton & Neil's TNormal, weighted mean, weighted min, weighted max aggregation functions. Reduces high-dimensional CPT elicitation to a handful of weights for monotonic-relationship nodes.
3. **Sensitivity analysis.** Morris screening (cheap, qualitative) and Sobol indices (expensive, quantitative) wrapping SALib. Identifies which CPT entries dominate output variation; prioritizes elicitation effort.

All three are fully unit-testable and have no external dependencies beyond numpy and SALib.

**Deliverables.**

- `src/elicitation/engine/aggregation.py` — `linear_pool`, `logarithmic_pool`, `cooke_pool`.
- `src/elicitation/engine/ranked_nodes.py` — TNormal, weighted aggregation functions, CPT generator from ranked-node parameters.
- `src/elicitation/engine/sensitivity.py` — `morris_screening`, `sobol_indices` (wrapping SALib), per-CPT-entry influence rankings.
- `tests/elicitation/test_engine.py` — unit coverage of each primitive against published reference outputs.

**Validation.**

- Linear and logarithmic pooling agree on degenerate inputs (single expert) and produce expected outputs on the simple two-expert symmetric/asymmetric cases.
- Cooke weighted aggregation reproduces published examples from Cooke's classical-model textbook against the same seed questions.
- Ranked-node CPT generation reproduces the Fenton & Neil 2007 paper's published examples.
- Morris and Sobol outputs agree with SALib's reference outputs on a small test BN.

### Layer 2 — Protocol implementations: Cooke / IDEA / SHELF

**Status.** ⬜ not started
**Resolves.** Diagnosis item 8 (no support for multiple protocols).

**Latent-regime impact.** Under the Plan 1 latent-regime topology (see `docs/01_latent_regime_plan.md`), the CPTs the protocols elicit change shape entirely. The old labelling CPT $P(S \mid D, T, P)$ is removed; the new emission CPTs $P(D \mid S, \ldots)$, $P(T \mid S, \ldots)$, $P(P \mid S, \ldots)$ are elicited from scratch, plus a regime prior $\pi(S)$. The elicitation questions become generative ("given the regime, what does damage look like?") rather than labelling ("given outcomes, which regime?"), which is easier to defend with domain experts. The `CPTColumnTarget` shape covers all of these; Layer 2 is topology-agnostic at the protocol level. The deployment configuration (Layer 0 YAML) names which CPTs are in scope for elicitation given the active topology.

**Scope.** Three configurable workflows, each implemented as a state machine persisted in `elicitation_session_events`. Resumable across browser sessions. Each protocol exposes:

- `required_experts() -> tuple[int, int]` — min/max expert count.
- `workflow() -> WorkflowSpec` — the ordered sequence of steps.
- `aggregate(expert_inputs) -> CPTColumn` — protocol-specific aggregation.
- `provenance_record() -> ProvenanceMetadata` — what gets written to `cpt_provenance` at conclusion.

The three protocols differ along three axes (number of experts, per-expert workflow, aggregation method) but share the same state-machine infrastructure.

- **`SHELFProtocol`.** Single expert (or small group). Quantile-based elicitation (5th, 50th, 95th percentiles per parameter). Roulette method or interactive distribution-fitting. Aggregation: identity for single expert, linear pool for small group. Best for moderate-stakes contexts and solo analyst work.
- **`IDEAProtocol`.** 3-7 experts. Two-round iterative: private estimate → group discussion → private revised estimate → aggregation. Aggregation: linear or geometric pooling, selectable. Best for mid-stakes contexts with multi-analyst teams.
- **`CookeProtocol`.** 4-12 experts. Calibration questions answered first; performance scores compute per-expert Cooke weights; target questions then weighted by calibration. Best for high-stakes regulatory contexts. Requires a deployment-specific calibration question set in the domain.

Each protocol's `workflow()` is parameterised by an `ElicitationTarget`. The default target is a single CPT column (one $P(Y \mid \text{Pa}(Y) = u)$ at a time). One additional target shape is supported so the protocols cover the elicitation surfaces that Plans 3 and 5 require:

- **`CPTColumnTarget`** — default. Elicits one row of a CPT. Shape: a categorical distribution on $|Y|$ states.
- **`RankedNodeTarget`** — for Layer 5's visual ranked-node UI. Elicits the per-parent weights and aggregation function of a Fenton & Neil ranked node; the CPT is generated downstream from these weights.

Both targets share the same state-machine infrastructure; they differ in the per-step UI components (Layer 3) and in the validation that the aggregated output is well-formed (CPT row vs ranked-node parameter set).

A temporal `TransitionMatrixTarget` (row-stochastic transition matrices for an HMM extension) is **not in scope** because BN↔HMM integration is out of scope for all five plans; see `docs/master_plan.md` §6 (Gaps) and `docs/bn_hmm_integration.md`. If a temporal BN extension is eventually built, the `ElicitationTarget` abstraction is the right hook to add it without restructuring Layer 2.

**Deliverables.**

- `src/elicitation/protocols/base.py` — `ElicitationProtocol` abstract class, `WorkflowSpec`, `ProvenanceMetadata`, `ElicitationTarget` base class.
- `src/elicitation/protocols/targets.py` — `CPTColumnTarget`, `RankedNodeTarget` and their validation/aggregation hooks.
- `src/elicitation/protocols/shelf.py` — `SHELFProtocol` implementation.
- `src/elicitation/protocols/idea.py` — `IDEAProtocol` implementation.
- `src/elicitation/protocols/cooke.py` — `CookeProtocol` implementation, with calibration-question scoring helpers.
- `tests/elicitation/test_protocols.py` — end-to-end protocol runs against synthetic expert inputs.

**Validation.**

- Each protocol runs end-to-end on a test network, producing a CPT with attached provenance.
- State machine persists across simulated restarts (kill mid-workflow, resume from the database).
- Aggregation outputs match the published methodology references for canonical small examples.
- Cooke's protocol correctly down-weights an expert who fails the calibration questions and matches reference outputs from the classical-model literature.

### Layer 3 — UI layer (Streamlit)

**Status.** ⬜ not started
**Resolves.** Makes Layers 0–2 usable by non-engineer analysts and domain experts.

**Scope.** Streamlit pages exposing the elicitation workflows. Streamlit-first because it is the fastest path to a usable interactive UI for v1. Replace with a dedicated frontend (React or similar) when scaling demand or feature complexity justifies the investment.

Pages and tabs:

- **New CPT elicitation.** Pick a node, pick a protocol, run the wizard step-by-step. Workflow state persists between page loads.
- **CPT review and override.** Inspect a current CPT, compare to historical versions, override a single cell with a manual entry (records the override in `cpt_provenance` as a manual edit).
- **Sources tab.** Shared with translator workflow from Plan 2's B1b. Single source-credibility registry serves both translation and elicitation contexts.
- **HITL triage.** Shared with translator workflow from Plan 2's E1. Confidence-driven review queue serves both translator outputs and elicitation proposals.
- **Calibration dashboard.** Per-expert Cooke weight history; per-CPT calibration over time (where outcome data exists); model-level calibration plots.
- **CPT version history viewer.** Time-machine view of any CPT across all elicitation sessions that produced it.

**Deliverables.**

- `app/elicitation/` — Streamlit pages.
- `app/elicitation/components/` — shared UI components (distribution editors, quantile pickers, calibration plots, version diff views).
- `app/elicitation/styles.css` — separate stylesheet, loaded once.

**Validation.**

- Each protocol workflow runs end-to-end in the Streamlit UI.
- Auth and per-deployment isolation work correctly (login screens, session management, no cross-deployment data leakage).
- All visualisations render correctly across the supported browsers.
- Walkthrough documentation exists for each workflow (`docs/elicitation_walkthroughs.md`).

### Layer 4 — Integration with inference

**Status.** ⬜ not started
**Resolves.** Diagnosis item 9 (no reuse across engagements). Couples Plan 4 to Plan 3's inference engine.

**Scope.** Elicited CPTs export to Plan 3's `NetworkSpec`. Round-trip: elicit → save → load into `PgmpyBackend` or `PymcBackend` → run inference. Per-CPT $\kappa$ values from the elicitation provenance are respected by `PymcBackend`'s Dirichlet priors (closes M3 from the dashboard review for the elicited-CPT path).

The export interface:

```python
# src/elicitation/export/network_spec.py
def cpts_to_network_spec(
    network_id: int,
    snapshot_at: datetime | None = None,
) -> NetworkSpec:
    """Build a NetworkSpec from this deployment's CPTs.
    
    If snapshot_at is given, uses the CPT versions in force at that time;
    otherwise uses current versions.
    """
```

The inverse direction (`network_spec_to_cpts`) is also supported, so that an existing `NetworkSpec` (e.g., the bootstrap Hormuz network from Plan 3 Phase 0) can be imported into the elicitation store as the starting point for refinement.

**Deliverables.**

- `src/elicitation/export/network_spec.py` — `cpts_to_network_spec` and `network_spec_to_cpts`.
- `src/elicitation/export/cli.py` — command-line tools for export/import (useful for CI pipelines and customer migrations).
- Integration tests exercising the round trip.

**Validation.**

- Round-trip: load Hormuz from `src/network.py` → store in elicitation DB → export back → resulting `NetworkSpec` is identical.
- Per-CPT $\kappa$ values from `cpt_provenance` flow through to `PymcBackend`'s Dirichlet priors.
- Snapshot-at-time queries return the historically-correct CPT version.
- Both backends consume elicited CPTs and produce posteriors matching the current-model baseline (within MCMC error for PymcBackend).

### Layer 5 — Advanced features

**Status.** ⬜ not started
**Resolves.** Diagnosis items 1 (hardcoded CPTs replaced by elicited, calibrated CPTs), 2 (multi-expert aggregation now first-class), 3 (calibration tracking now infrastructure), 10 (LLM corpus coupling).

**Scope.** Five advanced subsystems on top of the v1 platform:

1. **LLM-proposed initial CPT values.** Couples to Plan 2's E2 (RAG memory). For a new CPT being elicited, the LLM retrieves the most relevant analog historical events from the translator audit log and proposes initial CPT values with span-grounded citations. The expert reviews, edits, or rejects. The proposal step does not commit anything; it gives the human elicitor a starting point.
2. **Ranked-node visual UI.** The Fenton & Neil ranked-node methodology from Layer 1 gets a visual elicitation surface: instead of entering CPT cells, the expert specifies per-parent weights and an aggregation function (TNormal, weighted mean, etc.). The Streamlit UI generates the full CPT from these inputs and shows it to the expert for review.
3. **Sensitivity-driven prioritization workflow.** Wraps Layer 1's Morris/Sobol primitives in an analyst-facing workflow: "show me which CPT entries dominate the scenario posterior under the current evidence." The output is a ranked list of CPT cells to elicit formally; the rest can stay as analyst placeholders.
4. **Calibration Tier 2 — intermediate-node tracking.** For nodes where outcomes can be observed (e.g., "did `Tanker_Incidents = frequent` actually materialise in the month following the model's prediction?"), record the realised outcome in `outcomes` and compute Brier scores and calibration plots over time. Refines per-CPT $\kappa$ values via empirical updating.
5. **Calibration Tier 3 — Bayes factor / regime trajectory.** For the latent-regime model (Plan 1), record log-Bayes-factor predictions for each evidence increment against expert-judged "true" regime trajectories on historical analog events. Reveals which CPT regions are systematically miscalibrated.

Subsystem 4 also feeds back into the Cooke protocol from Layer 2: experts who participate in CPT elicitation accrue calibration scores over time as their contributions' outcomes are observed, and their Cooke weights in future elicitations update accordingly.

**Deliverables.**

- `src/elicitation/proposals/llm.py` — RAG-augmented CPT proposal generator (consumes the translator's E2 index from Plan 2).
- `src/elicitation/ranked_nodes/ui.py` — Streamlit components for ranked-node visual elicitation.
- `src/elicitation/sensitivity/workflow.py` — analyst-facing prioritization workflow.
- `src/elicitation/calibration/tier2.py` — intermediate-node outcome tracking and Brier scoring.
- `src/elicitation/calibration/tier3.py` — Bayes factor calibration against historical trajectories.
- `src/elicitation/calibration/expert_weights.py` — Cooke weight updates from accrued calibration data.

**Validation.**

- LLM proposals are reviewable in the Streamlit UI; analyst can accept, edit, or reject; the choice is logged.
- Ranked-node UI generates CPTs matching Layer 1's pure-function output on the same inputs.
- Sensitivity workflow ranks CPT entries in agreement with Sobol indices on a test network.
- Tier 2 calibration data accumulates over a simulated history; Brier scores and calibration plots render correctly.
- Tier 3 historical replays produce Bayes-factor trajectories that match expert-judged "true" trajectories on the historical analog events (within calibrated noise).
- Cooke weights for experts update sensibly as their accrued predictions are scored against outcomes.

### Layer 5.5 — Agentic AI experts (calibration-validated)

**Status.** ⬜ not started — extension of Layer 5; optional. Ships only after the human protocols (Layer 2) and the calibration substrate (Layers 0 and 5) exist, because it *reuses* them.
**Resolves.** Extends diagnosis items 1, 2, 3, 10 to the case where some or all "experts" are LLM agents — to accelerate and scale elicitation **without** sacrificing defensibility.

**Positioning — "calibration-validated AI elicitation," not "AI replaces experts."** Layer 5.1 already uses an LLM as a *proposer*. This layer lets LLM agents act as *experts inside the protocols*, but strictly as far as their **measured calibration** licenses. The enabler is that Cooke's seed-question scoring (Layer 2) is itself a calibration test: an LLM "expert" answers the same known-answer seeds a human would, is scored identically (statistical accuracy × information), and is weighted — or zeroed below the cutoff — on that evidence. This turns "should we trust the AI's numbers?" from faith into a measured, auditable quantity. (Evidence base in Section E.2.)

**Three modes (increasing autonomy, decreasing stakes-appropriateness):**

1. **Proposer** (already Layer 5.1). LLM drafts CPTs via RAG; humans are the experts and sign off. Any stakes.
2. **One scored voting member.** An LLM agent joins a human IDEA/Cooke panel as one more `expert` row with its own `expert_calibration` record from the seeds; its weight is its measured performance. Humans retain sign-off. Mid-stakes, or high-stakes as an adjunct.
3. **Fully automated panel** (fast/cheap tier + red-team). A panel of *diverse* LLM agents runs an entire IDEA or Cooke workflow end-to-end (estimate → discuss → revise → aggregate, or seed-score → weight → pool), emitting a CPT plus a calibration report with no human in the loop. For low/medium-stakes bulk CPT generation and as an adversarial red-team that widens human panels' blind spots — **not** for regulatory high-stakes without human review.

**The correlation problem (handled, not hidden).** LLM agents off the *same* base model are not independent — they share training data and biases, so naive pooling overstates confidence. Mitigations, all recorded in provenance: (i) compose panels from **genuinely different base models** (e.g., Claude + GPT + open-weights), not just personas/temperatures of one; (ii) estimate and report inter-agent correlation; (iii) shrink the effective sample size / down-weight accordingly. The empirical backbone: a *diverse 12-model ensemble* matched human crowds (Schoenegger et al. 2024) whereas a *single* GPT-4 did not (Schoenegger & Park 2023) — diversity is doing the work (Section E.2).

**Why the existing abstractions already fit.** An LLM agent is just another `expert` (Layer 0) with an `expert_calibration` record, driven through the same `ElicitationProtocol` state machine and `CookeProtocol` seed-scoring (Layer 2). No restructuring — this layer is purely additive.

**Deliverables.**

- `src/elicitation/agents/llm_expert.py` — `LLMExpert` adapter implementing the *same* per-step interface a human uses (answer a quantile question, revise after discussion, etc.).
- `src/elicitation/agents/panel.py` — multi-model panel orchestration (diverse providers); IDEA/Cooke runners over LLM experts.
- `src/elicitation/agents/decorrelation.py` — inter-agent correlation estimation and effective-weight adjustment.
- `src/elicitation/agents/calibration_report.py` — per-agent seed-calibration scores + panel-level report, written to `cpt_provenance`.
- `tests/elicitation/test_llm_experts.py` — seed-scoring on a known-answer set; a poorly-calibrated agent is down-weighted/zeroed; same-model correlation is detected.

**Validation.**

- An LLM expert scored on the deployment's seed set yields a calibration score; under **leave-one-seed-out cross-validation** its weighted contribution does not degrade the Decision Maker versus equal weight — the Clemen (2008) test (Section E.1), applied to AI experts.
- Same-model agents show high measured correlation; cross-model agents lower — and the effective weight adjusts accordingly.
- A fully-automated panel's CPT lands within a stated tolerance of a human-elicited reference CPT on a back-test set, with calibration reported.
- Every AI-sourced CPT is flagged in `cpt_provenance` with its calibration score, model set, and correlation note — a hard defensibility requirement.

### Layer 6 — Commercial layer (deferred)

**Status.** ⬜ not started — deferred until paying customers exist.
**Resolves.** Productisation: billing, onboarding, support.

**Scope.** Out of scope for the engineering work in Layers 0–5. Included here as a placeholder so the layered architecture is complete. Includes billing integration (Stripe or similar), customer onboarding flows, tenant-level configuration UI (selecting which protocols are available, which calibration tiers are enabled), documentation portal, support ticketing integration.

**Deliverables.** Deferred.

**Validation.** Deferred.

## Section B — Design decisions resolved

The decisions below are resolved.

1. **Platform positioning — Decided: methodology-as-product (Option B), open-core licensing.** Engine + protocols + math primitives are open-source under a permissive license. Commercial layer (deployment automation, hosted versions, premium integrations, support contracts) is closed. This matches standard B2B PPL-adjacent tooling and aids adoption while preserving monetisation.
2. **Deployment shape — Decided: multi-deployment, single-tenant per deployment.** Each customer engagement gets its own isolated stack. No `tenant_id` columns. No SaaS data co-mingling. Matches the high-stakes regulatory and consulting-led nature of target use cases.
3. **UI strategy — Decided: Streamlit for v1, upgrade later.** Streamlit forms are the fastest path to a usable interactive UI. Replace with a dedicated frontend (React or similar) when scaling demand or feature complexity justifies the investment. Do not over-engineer the UI before product-market fit.
4. **Storage — Decided: SQLite for development, Postgres for production.** SQLAlchemy / SQLModel ORM. Alembic migrations. Per-deployment isolated database.
5. **Audit log substrate — Decided: extend Plan 2's schema rather than duplicate.** The `articles`, `translations`, `analyst_actions`, and `sources` tables from Plan 2 are extended with elicitation-adjacent join tables rather than re-created. Single source of truth for shared concepts (sources, analyst actions, audit events).
6. **Protocol coverage — Decided: Cooke, IDEA, SHELF all in Layer 2.** Three protocols, one platform, configurable at elicitation start. Cooke for high-stakes regulatory contexts (Hormuz-style); IDEA for mid-stakes corporate decisions; SHELF for solo analyst work.
7. **CPT compression — Decided: ranked nodes (Fenton & Neil) as the primary compression method.** Implemented in Layer 1, exposed as a visual elicitation surface in Layer 5. Noisy-OR / Noisy-MAX deprioritised because most relationships in Hormuz-style models are non-additive.
8. **Calibration tiers — Decided: three tiers with phased introduction.** Tier 1 (translator-level) shipped via Plan 2's D2. Tier 2 (intermediate-node) infrastructure in Layer 5 from day one; signal accumulates over months. Tier 3 (Bayes factor / regime trajectory) deferred until the latent regime (Plan 1) is in production.
9. **Translator coupling — Decided: deep.** The translator's audit log (Plan 2 D3) is the source of analog historical events for anchored elicitation; the translator's RAG memory (Plan 2 E2) feeds LLM-proposed CPT values; the translator's HITL queue (Plan 2 E1) serves elicitation proposals.
10. **Cooke calibration question set — Decided: per-deployment.** Each customer engagement constructs its own domain-relevant calibration question set during onboarding. Re-used across all Cooke protocol runs in that deployment. Mostly relevant for the Hormuz reference deployment in the near term; corporate deployments using IDEA or SHELF do not require it.
11. **Agentic AI experts — Decided: calibration-validated, additive, attributable (Layer 5.5).** LLM agents may act as experts *only* via the same seed-scoring as humans; panels must be **multi-model** (decorrelation), every AI-sourced CPT is flagged in `cpt_provenance` with its calibration score, model set, and correlation note, and **human sign-off is required for high-stakes** work. Positioning is "calibration-validated AI elicitation," not replacement. The methodology rests on the peer-reviewed work catalogued in Section E.

## Section C — Open questions

These do not block Layer 0 but should be resolved before the corresponding layer begins:

| Question | Block | Notes |
| --- | --- | --- |
| Per-deployment Postgres setup automation | Layer 0 | Docker Compose for the engineering MVP; Terraform / Helm for production. Choose toolchain. |
| Auth provider for SSO integration | Layer 0 | Self-hosted (Keycloak, Authentik) vs hosted (Auth0, Clerk). Recommend hosted for v1 to ship faster. |
| LLM model for proposal generation | Layer 5 | Anthropic API (Claude) or OpenAI? Coupled to Plan 2's translator provider choice. |
| AI-panel model diversity & correlation handling | Layer 5.5 | How many distinct base models, and how to estimate/adjust for inter-agent correlation. Backed by Schoenegger et al. 2024 vs Schoenegger & Park 2023 (Section E.2). |
| Leakage-free seed set for AI calibration | Layer 5.5 / Layer 2 | LLM training-data contamination can inflate seed scores; need rolling, post-cutoff seeds (ForecastBench-style; Section E.2) so AI-expert calibration can't be gamed. |
| Cooke calibration question set design | Layer 2 | Hormuz-specific seed questions need authoring with domain experts. Bootstrapping cost. Defer until Layer 2 ships and the first Cooke deployment is concrete. |
| UI upgrade trigger | Layer 3 | When does Streamlit stop being sufficient? Define the criterion (e.g., > N concurrent users per deployment, > M custom UI components needed). |
| Open-source license choice | Layer 0 | MIT, Apache 2.0, BSD-3? Recommend Apache 2.0 for patent grant; compatible with open-core commercial layer. |

## Section D — Execution order summary table

For coherence with the format used in Plans 2 and 3:

| Order | Layer | Resolves | Rationale |
| --- | --- | --- | --- |
| 1 | 0 — Data model and storage substrate | Foundation | Per-deployment isolation, versioning schema, auth scaffolding. Unblocks every subsequent layer. Extends rather than duplicates Plan 2's schema. |
| 2 | 1 — Core engine | Diagnosis items 6, 7 | Aggregation primitives, ranked nodes, sensitivity analysis. Pure-function library; no UI or DB dependencies; fully unit-testable. |
| 3 | 2 — Protocol implementations | Diagnosis item 8 | Cooke / IDEA / SHELF as state-machine workflows. Configurable at elicitation start. The methodological core of the platform. |
| 4 | 3 — UI layer (Streamlit) | Usability | Makes Layers 0–2 accessible to analysts and domain experts. v1 milestone: end-to-end elicitation usable by non-engineers. |
| 5 | 4 — Integration with inference | Diagnosis item 9 | Elicited CPTs → `NetworkSpec` → `PgmpyBackend` / `PymcBackend`. Round-trip with Plan 3's inference layer. |
| 6 | 5 — Advanced features | Diagnosis items 1, 2, 3, 10 | LLM-proposed CPTs, ranked-node UI, sensitivity prioritization, calibration tiers 2 and 3, Cooke weight updates from outcomes. |
| 6b | 5.5 — Agentic AI experts (extension, optional) | Items 1, 2, 3, 10 (AI-assisted) | LLM agents as calibration-scored experts; multi-model panels with decorrelation; provenance-flagged; human sign-off for high-stakes. Additive to Layer 5. |
| 7 | 6 — Commercial layer | Productisation | Deferred until paying customers exist. Billing, onboarding, support, tenant config UI. |

## Section E — Related work and methodology provenance

All references below were verified against primary sources (focused search, June 2026). Each entry states what the work is, **how it supports** this plan's methodology, and **where it points to further research or stronger validation**. This section is the evidence base for Layers 2 (protocols), 5 (calibration), and 5.5 (agentic AI experts).

### E.1 Structured expert judgment — the human protocols and their validity debate

- **Aspinall (2010), *A route to more tractable expert advice*, Nature 463:294–295.** Opinion piece arguing for Cooke's performance-weighted ("classical model") elicitation; documents the Montserrat volcano crisis where a weighted expert panel produced usable guidance within ~2 hours.
  - *Supports:* the case that structured, performance-weighted elicitation is decision-grade in high-stakes settings — the rationale for Cooke being Layer 2's regulatory-tier protocol.
  - *Research/validation:* instrument our tooling to measure wall-clock from question to defensible CPT, and reproduce the "fast, usable guidance" property as a product metric.

- **Hemming, Burgman, Hanea, McBride & Wintle (2018), *A practical guide to structured expert elicitation using the IDEA protocol*, Methods in Ecology and Evolution 9(1):169–180.** Step-by-step IDEA: investigate → individual estimates **with credible intervals** → feedback vs other experts → discuss → private revision → aggregate.
  - *Supports:* the concrete IDEA state machine in Layer 2, including the interval-feedback step.
  - *Research/validation:* adopt their interval-rescaling (extrapolate each expert's stated interval to a fixed 90% interval) as an explicit Layer 1 debiasing primitive, and A/B test its effect on our elicited CPTs.

- **SHELF — Gosling (2018), *SHELF: The Sheffield Elicitation Framework* (in *Elicitation*, Springer); O'Hagan & Oakley, `SHELF` R package (CRAN).** Behavioural aggregation: a facilitator guides the group to a single consensus ("rational impartial observer") distribution; roulette and quantile elicitation modes.
  - *Supports:* the SHELF protocol and its roulette/quantile UIs in Layers 2–3.
  - *Research/validation:* use the CRAN `SHELF` distribution-fitting routines as a reference oracle to cross-check our Layer 1 fitters.

- **Clemen (2008), *Comment on Cooke's classical method*, Reliability Engineering & System Safety 93(5):760–765**, with the defense in **Colson & Cooke (2018), *Expert Elicitation: Using the Classical Model to Validate Experts' Judgments*, Review of Environmental Economics and Policy 12(1):113–132.** The active cross-validation debate: does performance-weighting actually beat equal-weighting *out of sample*?
  - *Supports:* makes leave-one-seed-out **cross-validation a first-class acceptance test** in our calibration layer, not an afterthought — and tells us to report performance-weight vs equal-weight head-to-head per deployment.
  - *Research/validation:* implement leave-one-seed-out scoring (Layer 5) for both human and (Layer 5.5) AI panels; this is exactly the test Layer 5.5's validation invokes.

### E.2 LLM forecasting and calibration — the basis for AI experts

- **Kadavath et al. (2022), *Language Models (Mostly) Know What They Know*, arXiv:2207.05221 (Anthropic).** Larger LMs are reasonably calibrated on multiple-choice / true-false when formatted correctly, and can partly self-predict whether they know an answer.
  - *Supports:* the premise that an LLM's probabilities are *measurable and sometimes usable* — the precondition for seed-scoring an LLM expert (Layer 5.5).
  - *Research/validation:* the "mostly" is the catch — measure our agents' calibration on **domain** seeds, not generic benchmarks, before granting them any weight.

- **Lin, Hilton & Evans (2022), *Teaching Models to Express Their Uncertainty in Words*, arXiv:2205.14334.** Proof of concept that a model can emit **calibrated verbalized confidence** (e.g. "90% confident") on a task (CalibratedMath).
  - *Supports:* lets an LLM expert answer the *same* verbalized-quantile questions a human does in SHELF/IDEA/Cooke.
  - *Research/validation:* compare verbalized vs logit-derived confidence for our categorical CPT columns; fit and recalibrate the agent's stated quantiles.

- **Halawi, Zhang, Yueh-Han & Steinhardt (2024), *Approaching Human-Level Forecasting with Language Models*, arXiv:2402.18563.** A retrieval-augmented LM forecasting system that, on post-cutoff questions, **nears (and sometimes beats) the competitive-human crowd aggregate**.
  - *Supports:* the RAG-augmented proposer (Layer 5.1) and the viability of LLM experts — *provided they retrieve, not merely recall*.
  - *Research/validation:* wire agent elicitation to Plan 2's RAG memory (E2) and test whether retrieval measurably improves seed-calibration.

- **Schoenegger et al. (2024), *Wisdom of the Silicon Crowd*, Science Advances (arXiv:2402.19379)** and **Schoenegger & Park (2023), *Large Language Model Prediction Capabilities*, arXiv:2310.13014.** A **diverse 12-model ensemble** matched a 925-person human crowd; a **single GPT-4** was no better than chance (50%-everything).
  - *Supports:* the **multi-model-diversity requirement** for AI panels (Layer 5.5) — the empirical backbone of the correlated-error argument.
  - *Research/validation:* quantify the diversity↔accuracy trade-off on our domain; find the minimum viable model-diversity for a calibrated panel.

- **Karger et al. (2024), *ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities*, arXiv:2409.19839 (ICLR 2025).** Leakage-free, continuously-updated forecasting benchmark; on a held-out subset **expert humans still beat the best LLM (p = 0.01)**.
  - *Supports:* honest positioning — AI experts are an **adjunct, not a replacement**, especially high-stakes (Layer 5.5 mode limits).
  - *Research/validation:* adopt a ForecastBench-style **rolling, post-cutoff seed set** per deployment so AI-expert calibration cannot be inflated by training-data contamination.

### E.3 Multi-agent LLMs and LLM-elicited priors

- **Du, Li, Torralba, Tenenbaum & Mordatch (2023), *Improving Factuality and Reasoning in Language Models through Multiagent Debate*, arXiv:2305.14325 (ICML 2024).** Multiple LLM instances propose and **debate over rounds**, improving reasoning/factuality and reducing hallucination.
  - *Supports:* the "discuss" phase of an automated IDEA panel (Layer 5.5 mode 3).
  - *Research/validation:* test whether debate genuinely *adds information* or merely amplifies the shared-prior (correlation) failure — measure post-debate calibration against round-1. (This paper reports gains; the correlation risk is our open question, not its finding.)

- **Capstick, Krishnan & Barnaghi (2024), *AutoElicit: Using Large Language Models for Expert Prior Elicitation in Predictive Modelling*, arXiv:2411.17284 (ICML 2025).** LLMs produce **Gaussian priors (mean + sd) for linear-model parameters**; priors are informative, natural-language-refinable, and beat in-context learning (case study saved ~6 months of labelling).
  - *Supports:* the closest published precedent for "LLM → Bayesian prior"; validates both the proposer (Layer 5.1) and the (θ̂, κ) → Dirichlet-prior path (Layer 4).
  - *Research/validation:* **extend AutoElicit's linear-Gaussian result to categorical CPT columns / Dirichlet priors** — our actual target, and an open, publishable step.

### E.4 The gap this plan occupies

A focused June 2026 search did not surface a standardized, published method that **marries Cooke's classical model (seed-validated, performance-weighted, poor experts zeroed) to an *LLM* expert panel feeding a Bayesian network's CPTs.** The components above are validated *separately* — structured expert judgment (E.1), LLM forecasting/calibration (E.2), multi-agent aggregation and LLM-elicited priors (E.3). Layer 5.5 assembles them. This is both the **differentiation** (a defensible, calibration-validated AI-elicitation product) and a genuine **research contribution** (we cannot prove the negative, but the configuration appears under-occupied). It should be validated with the same rigour the SEJ literature demands: domain seed sets, leave-one-seed-out cross-validation, and explicit correlation accounting.

---

**End of plan.** Companion plans: `docs/02_translator_robustification.md` (Plan 2, evidence ingestion) and `docs/03_pymc_integration_plan.md` (Plan 3, inference engine). Math context: `docs/01_latent_regime_plan.md` Section A.8. Methodology provenance: Section E above.
