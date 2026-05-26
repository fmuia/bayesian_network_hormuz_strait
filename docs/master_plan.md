# Master Plan: From Demo to Productisable Platform

> **Status.** Living document, last updated 2026-05-26 (review-driven fixes: decision-log numbering, finding-count arithmetic, M3/M4/M7 mapping corrections, deferred-decisions table, latent-regime cross-cuts surfaced in subplans, README touchpoints corrected, Plan 1 B1b re-slotted, Plan 2 Phase 6 removed and BN↔HMM workstream made entirely out of scope, continuous-viz UI ownership re-homed to Plan 4).
>
> **Purpose.** This is the orchestrator. It states the product vision, summarises the four sequential plans, maps every diagnosed finding to its plan, inventories all documentation in the repository, and records the strategic decisions made across the planning conversations. Start here.
>
> **Status legend.** ⬜ not started · ⏳ in progress · ✅ shipped (with date).

## Table of contents

1. [Vision](#vision)
2. [Architecture overview](#architecture-overview)
3. [Plan sequence: Step 0 + four plans](#plan-sequence)
4. [Findings coverage matrix](#findings-coverage-matrix)
5. [Existing roadmap items mapping](#existing-roadmap-items-mapping)
6. [Gaps and explicit non-goals](#gaps-and-explicit-non-goals)
7. [Repository documentation inventory](#repository-documentation-inventory)
8. [Decision log](#decision-log)

---



## 1. Vision

This repository contains the working code and documentation for a **methodology-as-product platform for Bayesian-network scenario modelling under deep uncertainty**. The current Hormuz network is the reference implementation; the long-term aim is a platform on which similar scenario models can be built and operated for clients in regulatory, intelligence, and corporate-strategy contexts where rare-tail outcomes matter and where data-driven model fitting is impossible.

The platform consists of four layered concerns:

- **Evidence ingestion** — converting unstructured input (news headlines, articles, analyst notes) into structured BN evidence with provenance, calibration, and audit trails.
- **Inference engine** — exact or sampling-based inference over Bayesian networks with discrete, continuous, or mixed nodes; hierarchical priors over CPTs; temporal extension for regime-tracking.
- **Methodology layer** — multi-protocol CPT elicitation (Cooke, IDEA, SHELF), versioned artefacts with full provenance, sensitivity-driven prioritisation, multi-expert aggregation, and calibration tracking.
- **Dashboard and UI** — the analyst-facing surface that exposes the three backend layers to stakeholders, with interactive visualisation, scenario exploration, audit views, and HITL review.

**Product positioning.** Methodology-as-product with open-core licensing. The inference engine, mathematical primitives, and protocol implementations are open-source under a permissive license; the commercial layer (deployment automation, hosted versions, premium integrations, support) is closed. Deployment shape is **multi-deployment, single-tenant per engagement** — each customer engagement gets its own isolated stack rather than sharing infrastructure SaaS-style. The Hormuz implementation is the reference engagement; methodology and code reuse across engagements is the productisation story.

**Differentiated against:** academic Bayesian-network tools (AgenaRisk, Netica, Hugin, GeNIe), which are general-purpose and don't ship with elicitation methodology or calibration tracking; commercial intelligence platforms (Recorded Future, RANE, Maplecroft), which are productised but methodologically opaque; forecasting platforms (Good Judgment Inc., Hypermind), which are calibration-strong but not BN-focused.

---



## 2. Architecture overview

```
┌──────────────────────────────────────────────────────────────────────┐
│ Dashboard / UI Layer    (Plan 4 — dashboard_ui_plan.md)              │
│ - Streamlit-first; modular component library                          │
│ - Backend-agnostic via the Posterior interface                        │
└──────────────────────────────────────────────────────────────────────┘
        ▲                ▲                    ▲                ▲
        │                │                    │                │
┌───────┴──────┐  ┌──────┴───────┐  ┌─────────┴──────┐  ┌──────┴─────┐
│ Evidence     │  │ Inference    │  │ Methodology    │  │ Shared     │
│ ingestion    │  │ engine       │  │ layer          │  │ substrate  │
│ (Plan 1)     │  │ (Plan 2)     │  │ (Plan 3)       │  │            │
│              │  │              │  │                │  │ audit log  │
│ translator,  │  │ pgmpy +      │  │ Cooke / IDEA / │  │ HITL queue │
│ ensemble,    │  │ PyMC dual    │  │ SHELF protocols│  │ versioning │
│ HITL review, │  │ backend with │  │ multi-expert   │  │ sources    │
│ provenance,  │  │ NetworkSpec, │  │ aggregation,   │  │ calibration│
│ RAG memory   │  │ continuous,  │  │ ranked nodes,  │  │ provenance │
│              │  │ HMM          │  │ sensitivity    │  │            │
└──────────────┘  └──────────────┘  └────────────────┘  └────────────┘
```

The arrows in the diagram represent runtime data flow:

- Plan 1 produces structured evidence; Plan 2 consumes it.
- Plan 3 produces calibrated CPTs; Plan 2 consumes them via `NetworkSpec`.
- Plan 2 produces posteriors; Plan 4 visualises them.
- Plans 1 and 3 share the audit log, HITL queue, and source-credibility registry; Plan 1 builds these, Plan 3 extends them.

The plans run **sequentially** (1 → 2 → 3 → 4) with internal items in each plan having their own ordering. Sequencing is by dependency, not just convenience. Before the four engineering plans, a conceptual foundation step (Step 0) resolves the most consequential design decision the project has made: the latent-regime reframing.

---



## 3. Plan sequence: Step 0 + four plans

The full work programme has five entries: a preliminary conceptual step that all four engineering plans depend on, then the four sequential plans themselves.

### Step 0 — Conceptual foundation: the latent-regime reframing

**Status.** ✅ Decision resolved 2026-05. Engineering implementation deferred to Plan 2 Phase 3.

**Where it came from.** Finding M1 in `docs/dashboard_review_2026-05.md` identified that the `Scenario` node in the current Bayesian network functions mathematically as a *softmax classifier of three intermediate-outcome nodes*, not as a probabilistic outcome in its own right. The 27-column CPT $P(S \mid D, T, P)$ is near-deterministic at corner configurations and only soft when the parents disagree — the entropy fingerprint of a labelling function, not a generative model. The headline "Stress_Mitigates 42%" number that the dashboard reports is mathematically the expected value of this labelling function under the joint posterior of $(D, T, P)$, dressed in the language of probability. See Part 2 and Appendix B of `notes/latent_regime_math.md` for the entropy-based diagnostic.

**The reframe.** Invert the arrows: $S$ becomes a **latent root node** with an explicit prior $\pi(S)$, and $(D, T, P)$ become **emissions** of $S$, alongside their existing upstream causes (which remain as parents). Inference over $S$ becomes genuine Bayesian regime inference:

$$
P(S = s \mid E) \propto \pi(s) \cdot P(E \mid S = s)
$$

Computable as **three exact inferences** (one per regime with $S$ clamped) plus a Bayes-rule combine. Scenario probabilities are now real posteriors over a latent regime variable, not expected values of a labelling function. See Parts 3–5 of `notes/latent_regime_math.md` for the full derivation including a worked numerical example.

**Why this is foundational, not just a Plan 2 item.** The decision touches all four engineering plans:

- **Plan 1 (translator).** The likelihood-semantics decision in A1 has cleaner mathematical meaning under a latent-regime model — Bayes factors $P(E \mid S = s_1) / P(E \mid S = s_2)$ become first-class quantities that the translator's output naturally feeds.
- **Plan 2 (inference).** Phase 3 is the **engineering implementation** of this conceptual decision. The hierarchical-prior infrastructure for the latent regime is the same infrastructure that closes M3 (per-CPT $\kappa$) and M4 (correlated CPT uncertainty).
- **Plan 3 (elicitation).** The CPTs experts elicit change shape entirely. The current labelling CPT $P(S \mid D, T, P)$ goes away; new emission CPTs $P(D \mid S, \ldots)$, $P(T \mid S, \ldots)$, $P(P \mid S, \ldots)$ are elicited. The questions become generative ("given the regime, what does damage look like?") rather than labelling ("given outcomes, which regime?") — easier to elicit defensibly from domain experts.
- **Plan 4 (UI).** Bayes factors as first-class outputs change how the dashboard communicates evidence strength to stakeholders. Plan 4 C4 (rich observed-node panel) explicitly exposes the per-observation Bayes-factor contribution once Phase 3 lands; C7 (before/after delta) remains a percentage-point chip, with a natural extension to a Bayes-factor mode under the latent-regime branch.

**Why this is a preliminary step.** The four engineering plans cannot be coherently specified without committing to this conceptual reframing. Plan 2 Phase 3 has no specification without it; Plan 3 doesn't know which CPTs to elicit; Plan 4's observed-node panel doesn't know what to display; even Plan 1's translator output has different meaning. The decision must be (and is) made *before* the plans begin.

**Math foundation.** The full mathematical treatment, accessible to a learner, is in `notes/latent_regime_math.md`:

- **Parts 1–2** — Bayesian-network foundations from scratch; what arrows really mean; why the current `Scenario` node is classifier-like (with entropy analysis).
- **Parts 3–5** — the latent-regime alternative, its inference mechanics, a worked numerical example using actual CPT values, and the argument for why this still answers the client's "scenario probabilities as model outputs" requirement.
- **Part 6** — practical implications across the four plans.
- **Appendix A** — independence, conditional independence, and d-separation (with the rain/sprinkler/wet worked example).
- **Appendix B** — Shannon entropy (the diagnostic used in Part 2).
- **Appendix C** — uncertainty under the latent regime: how the existing Dirichlet machinery transfers, what new sources of uncertainty appear, and what new outputs (Bayes factors, regime-conditional forecasts) become possible.

**Implementation status.** Conceptual decision ✅ made. Engineering implementation ⬜ not started; lives in Plan 2 Phase 3 with cross-cutting touches in all four plans.

---

### Plan 1 — Evidence ingestion: `docs/translator_robustification.md`

The translator turns unstructured news into BN evidence. The current implementation is a 510-line demo; this plan converts it into a tool with calibrated uncertainty, span-grounded reasoning, multi-model cross-checks, versioned prompts, structured audit trails, and HITL review.

12 items across five categories (A–E), with B1 split into 2 execution slots (B1a + B1b) for a total of 13 execution slots:

- **Semantic foundations**: A1 (likelihood semantics fix — closes the M2 bug that mispositions every CI in the dashboard), A2 (schema hardening).
- **Input and reasoning**: B1a/B1b (article-level input, source-credibility registry), B2 (span-grounded structured reasoning), B3 (abstention).
- **Uncertainty quantification**: C1 (self-consistency ensemble), C2 (multi-model cross-check).
- **Governance and evaluation**: D1 (versioned prompts), D2 (golden set + continuous evaluation), D3 (provenance audit log).
- **Operational integration**: E1 (HITL review queue), E2 (retrieval-augmented translation).

This plan also builds the **shared substrate** (audit log, versioned-artefact pattern, HITL queue, source-credibility registry) that Plans 3 and 4 reuse.

### Plan 2 — Inference engine: `docs/pymc_integration_plan.md`

The current pgmpy backend handles purely discrete networks with point-estimate CPTs and bolt-on Dirichlet resampling. This plan introduces a dual-backend architecture (`PgmpyBackend` for fast discrete inference, `PymcBackend` for hierarchical priors and continuous variables). The declarative `NetworkSpec` is the common input.

6 phases (0–5):

- **Phase 0**: refactor to introduce `NetworkSpec`.
- **Phase 1**: wrap existing inference in `PgmpyBackend` with the uniform `Posterior` interface.
- **Phase 2**: build `PymcBackend` for discrete-only networks; validate parity.
- **Phase 3**: latent-regime restructure (closes M1).
- **Phase 4**: continuous variable support (closes M2 properly at the inference layer, plus M3 via per-CPT κ).
- **Phase 5**: migrate Oil_Price to continuous in production.

Temporal extensions to the BN (a Markov chain on the regime variable) and BN↔HMM integration are **out of scope** for this plan. They require a separately-trained inflation HMM that does not exist in this repository and are tracked in `docs/bn_hmm_integration.md` as a longer-horizon workstream; see [Gaps](#gaps-and-explicit-non-goals). Roadmap A1 (evidence accumulation) is therefore not delivered by the four plans.

Mathematical context for the migration lives in `notes/latent_regime_math.md` (Parts 1–6, Appendices A–C).

### Plan 3 — Methodology layer: `docs/elicitation_tool_plan.md`

CPTs in the current model are inline literals chosen by one author without protocol. This plan delivers a multi-protocol elicitation platform: Cooke's classical model for regulatory contexts, IDEA for corporate decisions, SHELF for solo analysts. Includes versioned CPTs with full provenance, multi-expert aggregation, ranked-node compression, sensitivity-driven prioritisation, three-tier calibration tracking, and LLM-proposed CPT initial values via retrieval from Plan 1's audit log.

Layers 0–5 in scope (6 layers); Layer 6 (commercial: billing, onboarding, support) deferred until paying customers exist:

- **Layer 0**: data model and storage substrate; extends Plan 1's audit schema.
- **Layer 1**: core engine (aggregation primitives, ranked nodes, sensitivity analysis).
- **Layer 2**: protocol implementations.
- **Layer 3**: Streamlit UI.
- **Layer 4**: integration with Plan 2's inference engine.
- **Layer 5**: advanced features (LLM-proposed CPTs, calibration tiers 2 and 3, Cooke weight updates from accrued outcomes).

### Plan 4 — Dashboard UI and polish: `docs/dashboard_ui_plan.md`

Closes the visualisation, dashboard-architecture, performance, code-hygiene, and test-coverage findings that the three backend plans leave open.

5 categories with 29 items:

- **Category A — architectural refactor**: split `dashboard.py`, extract CSS, fix engine caching, derive DAG levels topologically.
- **Category B — performance**: bound caches, memoise evolution series, apply-button override pattern.
- **Category C — visualisation improvements**: 14 items. C1–C13 address visualisation findings V2–V15 (V8 is handled by Category A's CSS extraction; V1 is covered by Plan 2 Phase 3). C14 owns the continuous Oil_Price panel and interval queries that Plan 2 Phase 4/5 enables — the work is in Plan 4 rather than Plan 2 so dashboard ownership stays in one place.
- **Category D — code hygiene**: dead-code triage, type-vocabulary cleanup, defensive-guard documentation, deduplication.
- **Category E — test coverage**: tests for state helpers, visualisation primitives, and dashboard components.

Several items in Categories A, D, and E are backend-independent and can be picked up opportunistically alongside the other plans.

---



## 4. Findings coverage matrix

This is the canonical mapping of every finding in `docs/dashboard_review_2026-05.md` to a plan. If a finding is not here, it is not covered anywhere — please raise it.

### Math findings (M*)


| ID  | Title                             | Plan                                                | Section                                                                               |
| --- | --------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------- |
| M1  | Scenario as classifier            | Step 0 (decision) + Plan 2 Phase 3 (implementation) | See Section 3 Step 0 and `notes/latent_regime_math.md`                                |
| M2  | Soft evidence semantics           | Plan 1 + Plan 2                                     | A1 + Phase 4                                                                          |
| M3  | Uniform κ                         | Plan 2 + Plan 3                                     | Phase 4 (per-CPT κ in `NetworkSpec`) + Layer 4 (per-CPT κ via elicitation provenance) |
| M4  | Independent CPT column resampling | Plan 2                                              | Phase 4 (hierarchical priors)                                                         |
| M5  | Material DAG omissions            | **Not in any plan**                                 | See [Gaps](#gaps-and-explicit-non-goals) below                                        |
| M6  | Root priors unjustified           | Plan 3                                              | Layer 2 (priors elicited via Cooke or IDEA)                                           |
| M7  | Resample-mean vs point-estimate   | Plan 2                                              | Phase 3 (resolved by latent-regime restructure)                                       |
| M8  | `+1e-6` Dirichlet guard           | Plan 4                                              | D3                                                                                    |
| M9  | Duplicate sensitivity functions   | Plan 4                                              | D4                                                                                    |


### Code findings (C*)


| ID  | Title                                       | Plan            | Section      |
| --- | ------------------------------------------- | --------------- | ------------ |
| C1  | `engine` cached but mutated                 | Plan 4          | A3           |
| C2  | `dashboard.py` 1878 lines                   | Plan 4          | A1           |
| C3  | Probability evolution recomputes everything | Plan 4          | B2 + B3      |
| C4  | Caches unbounded                            | Plan 4          | B1           |
| C5  | Soft-evidence mismatch (code side of M2)    | Plan 1 + Plan 2 | A1 + Phase 4 |
| C6  | Translator validation permissive            | Plan 1          | A2           |
| C7  | Silent renormalization                      | Plan 1          | A2           |
| C8  | Regex JSON extraction fragile               | Plan 1          | A2           |
| C9  | `render_network_png` dead code              | Plan 4          | D1           |
| C10 | `_NODE_LEVEL` hardcoded                     | Plan 4          | A4           |
| C11 | No `_merged_evidence` test                  | Plan 4          | E1           |
| C12 | `_PLUGINS_REGISTERED` workaround            | Plan 4          | D5           |
| C13 | `Observation.tone` unused                   | Plan 4          | D2           |
| C14 | No `viz.py` / dashboard helper tests        | Plan 4          | E2 + E3      |


### Visualisation findings (V*)


| ID  | Title                                   | Plan             | Section                                                 |
| --- | --------------------------------------- | ---------------- | ------------------------------------------------------- |
| V1  | Headline = resample mean                | Plan 2           | Phase 3 (resolves M7 which underlies V1)                |
| V2  | Info density wrong                      | Plan 4           | C1                                                      |
| V3  | Robustness thresholds arbitrary         | Plan 4           | C2                                                      |
| V4  | Slider UX trap                          | Plan 4           | C3                                                      |
| V5  | Hard-observed uninformative bars        | Plan 4           | C4                                                      |
| V6  | DAG layout crowded                      | Plan 4           | C5                                                      |
| V7  | Param vs forecast uncertainty conflated | Plan 4           | C6                                                      |
| V8  | CSS inline                              | Plan 4           | A2                                                      |
| V9  | No before/after on new observation      | Plan 4 + roadmap | C7 (delta-chip) + roadmap A3 (full waterfall)           |
| V10 | Latest-translation as plain text        | Plan 4           | C8                                                      |
| V11 | Colourblind-hostile palette             | Plan 4           | C9                                                      |
| V12 | Tooltip headline truncation             | Plan 4           | C10                                                     |
| V13 | Edge rationale isolated                 | Plan 4           | C11                                                     |
| V14 | Fixed narrative paragraphs              | Plan 4 + roadmap | C12 (hide for now) + roadmap B1 (responsive narratives) |
| V15 | DAG canvas height brittle               | Plan 4           | C13 (likely deferred)                                   |


### Summary

- **38 findings raised** in `docs/dashboard_review_2026-05.md` (9 M-findings, 14 C-findings, 15 V-findings).
- **37 findings have a home** in one of the four plans.
- **1 finding (M5)** is explicitly not covered; see [Gaps](#gaps-and-explicit-non-goals).

---



## 5. Existing roadmap items mapping (`docs/bn_app_next_steps.md`)

The pre-existing feature roadmap contains 11 items (A1–E1). Their relationship to the four plans:


| Roadmap item | Title                         | Status / Where it lives                                                                                                                                                                                      |
| ------------ | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A1           | Evidence accumulation         | **Not in the four plans.** The proper mechanism is a temporal Markov chain on the regime (or coupling to a separately-trained inflation HMM). Both require infrastructure that does not exist in this repository; see [Gaps](#gaps-and-explicit-non-goals) and `docs/bn_hmm_integration.md`. |
| A2           | Node-level credible intervals | ✅ Shipped 2026-04. Pre-dates this planning exercise.                                                                                                                                                         |
| A3           | Sensitivity attribution       | **Not in the four plans.** Compounds with Plan 4 C7 (delta chip) and Plan 2 Phase 3 (Bayes-factor decomposition makes this natural). Could be promoted to a Plan 4 item or kept as a roadmap item for later. |
| B1           | Daily narrative generation    | **Not in the four plans.** Compounds with Plan 4 C12 (responsive narratives). Requires LLM summary infrastructure beyond the translator.                                                                     |
| B2           | Pre-built scenario sequences  | **Not in the four plans.** Demo-time UX; trivial to add as a follow-on to Plan 4.                                                                                                                            |
| B3           | Session export                | **Not in the four plans.** `render_network_png` in Plan 4 D1 is annotated for B3's eventual use.                                                                                                             |
| C1           | Scenario comparison mode      | **Not in the four plans.** UI feature; compounds with Plan 4 information architecture.                                                                                                                       |
| C2           | CPT explorer                  | Delivered by Plan 3 (the elicitation tool is the CPT explorer, generalised).                                                                                                                                 |
| C3           | Undo/redo and pinning         | **Not in the four plans.** UI feature; would compound with Plan 4 state management.                                                                                                                          |
| D1           | Batch processing              | Delivered by Plan 1's article-level input and the multi-headline ingest path.                                                                                                                                |
| E1           | News memory database          | Delivered by Plan 1's E2 (RAG memory). The "two consumers, one memory" pattern (translator + narrative layer) is preserved.                                                                                  |


Roadmap items not delivered by the four plans (A3, B1, B2, B3, C1, C3) are kept in `docs/bn_app_next_steps.md` as a follow-on backlog. They are largely UI-flavoured features that compound naturally with Plan 4 work but aren't structural enough to warrant dedicated plan items.

---



## 6. Gaps and explicit non-goals

### Gaps acknowledged

- **M5 — Material DAG omissions.** The review identifies real causal channels not in the current DAG, most notably `Iran_Aligned_Militia_Attacks → US_Military_Response` (the Iraq/Syria base-strike pattern). This is a **modelling decision**, not an engineering one. It belongs in a future BN-structure review with domain experts, ideally using the elicitation tool from Plan 3 once it ships. Until then, the current DAG's scoping is documented in the edge-rationale tab and the `_EDGE_OMISSIONS` list.
- **Temporal BN extension and BN↔HMM integration (entirety of `docs/bn_hmm_integration.md`).** The integration doc describes four channels for coupling a BN to a data-trained inflation HMM: structural priors (Section 3.1.5), transition-matrix covariates (Approach A, Section 3.1), hierarchical sub-states (Approach B, Section 3.2), and emission modification (Approach C, Section 3.3). All four require a separately-trained inflation HMM that does not exist in this repository. A BN-internal temporal extension (a Markov chain on the regime variable) would be a precursor capability the BN needs before any of those couplings is possible; it too is out of scope for the four plans. Roadmap A1 (evidence accumulation) is the closest in-repo touchpoint and is therefore also not delivered. The whole workstream is tracked in `docs/bn_hmm_integration.md` for the longer horizon.

### Explicit non-goals

The following are **deliberately out of scope** for the four plans:

- **A full BN-structure re-elicitation.** The four plans take the current network's edge structure as given (modulo the latent-regime restructure in Plan 2 Phase 3). DAG-structure decisions are a separate workstream.
- **Multi-tenant SaaS infrastructure.** The platform is multi-deployment, single-tenant per engagement. No `tenant_id` columns, no row-level security. If a customer eventually demands SaaS, this is a retrofit, not a v1 concern.
- **Real-time news ingestion at scale.** The translator handles demo-cadence input (≤10 articles/day). High-throughput ingest is deferred until a customer demands it.
- **Mobile-first responsive design.** Stakeholder-facing committee tool; desktop-first is correct.
- **Internationalisation.** English-only for v1; the translator and elicitation prompts assume English source material.
- **Authentication and authorisation hardening beyond per-deployment basics.** SSO integration is in scope (Plan 3 Layer 0); fine-grained per-resource RBAC is not.

---



## 7. Repository documentation inventory

Every doc in this repository, in one place.

### Top-level

- `README.md` — quick-start for running the app and the BN-vs-HMM rationale. Touchpoints per plan: Plan 1 B1 updates the news-ingestion description (article-level input + source-credibility) and E2 adds the news-memory description; Plan 2 Phase 3 invalidates the current model-description section (latent-regime topology replaces labelling); Plan 2 Phase 5 adds the continuous Oil_Price section; Plan 3 Layer 3 adds an elicitation-tool quick-start; Plan 4 A1 updates any code-layout description. Reviewer should sweep the README at the end of each plan.

### `docs/`


| Doc                             | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `master_plan.md` (this doc)     | Orchestrator. Vision, sequencing, coverage matrix, decision log. Start here.                                                                                                                                                                                                                                                                                                                                                           |
| `dashboard_review_2026-05.md`   | Diagnostic review of the current dashboard. 38 findings (9 M-findings, 14 C-findings, 15 V-findings). Underlying motivation for the four plans.                                                                                                                                                                                                                                                                                        |
| `translator_robustification.md` | **Plan 1.** Evidence-ingestion layer. 12 items, A1–E2.                                                                                                                                                                                                                                                                                                                                                                                 |
| `pymc_integration_plan.md`      | **Plan 2.** Inference engine. 6 phases, 0–5.                                                                                                                                                                                                                                                                                                                                                                                           |
| `elicitation_tool_plan.md`      | **Plan 3.** Methodology layer. 6 active layers, 0–5 (Layer 6 deferred).                                                                                                                                                                                                                                                                                                                                                                |
| `dashboard_ui_plan.md`          | **Plan 4.** Dashboard UI and polish. 5 categories, 29 items.                                                                                                                                                                                                                                                                                                                                                                           |
| `bn_app_next_steps.md`          | Pre-existing feature roadmap. A1–E1 items. Several items are delivered by Plans 1–4; remaining items (A3, B1, B2, B3, C1, C3) are the follow-on backlog.                                                                                                                                                                                                                                                                               |
| `bn_hmm_integration.md`         | Longer-horizon BN↔HMM integration story (four channels: structural priors at Section 3.1.5; Approach A transition-matrix covariates at Section 3.1; Approach B hierarchical sub-states at Section 3.2; Approach C emission modification at Section 3.3). **Entirely out of scope for the four plans** — every channel needs a separately-trained inflation HMM that does not exist in this repository. Even a BN-internal temporal extension (a Markov chain on the regime variable, precursor to any of these) is out of scope. See [Gaps](#gaps-and-explicit-non-goals). |
| `model_documentation.md`        | Reference documentation for the current model: nodes, CPTs, inference mechanics.                                                                                                                                                                                                                                                                                                                                                       |


### `notes/` (gitignored, local-only)


| Doc                     | Purpose                                                                                                                                                                                                                                                                                                             |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `latent_regime_math.md` | Foundational math for Plan 2's latent-regime work. Parts 1–6 explain BNs from the ground up; Appendices A (independence/d-separation), B (Shannon entropy), C (uncertainty under the latent regime) cover supporting concepts. Local-only because it's a teaching artefact rather than a project decision document. |


---



## 8. Decision log

Strategic decisions resolved across the planning conversations, ordered roughly chronologically.

### Product strategy

1. **Position — Methodology-as-product (Option B), consulting-first evolving toward productised on-premise (Pattern 1).** The platform is sold as both implementation services and a licensable product. Each customer engagement is a bespoke deployment; the platform is the underlying tool that codifies the methodology.
2. **Licensing — Open core.** Engine, mathematical primitives, and protocol implementations are open-source under a permissive license (Apache 2.0 recommended). Commercial layer (deployment automation, hosted versions, premium integrations, support) is closed.
3. **Deployment shape — Multi-deployment, single-tenant per engagement.** No SaaS-style multi-tenancy. Each customer gets an isolated stack with its own database, app instance, users, and configuration.
4. **Initial deployment patterns — On-premise (Pattern 1) and managed single-tenant (Pattern 2).** Customer-hosted with support contracts, or hosted by us in the customer's cloud. Avoid pure SaaS for high-stakes regulatory use cases.

### Architecture

1. **Dual-backend with capability-based dispatch.** A declarative `NetworkSpec` describes the model; a dispatcher routes to `PgmpyBackend` (fast exact for discrete) or `PymcBackend` (hierarchical priors, continuous, HMM). Dashboard is backend-agnostic via a uniform `Posterior` interface.
2. **PyMC is mandatory when continuous variables are present.** All-discrete networks let the user choose; mixed networks must use PyMC.
3. **Shared substrate across Plans 1 and 3.** The audit log, versioned-artefact pattern, HITL queue, and source-credibility registry are built once in Plan 1 and extended in Plan 3. No duplication.
4. **Streamlit-first UI.** Streamlit forms and pages for v1. Upgrade to a dedicated web frontend (React or similar) when scaling demand or feature complexity justifies the investment.

### Methodology

1. **Likelihood semantics for translator output (Plan 1 A1).** The LLM produces likelihood ratios `ε_s = P(article | state=s) / max_s'`, not posterior-shaped distributions. Closes M2 cleanly.
2. **Three elicitation protocols supported (Plan 3 Layer 2).** Cooke's classical model for regulatory contexts (Hormuz-style); IDEA for corporate decisions; SHELF for solo analysts. Configurable at elicitation start.
3. **Ranked nodes as the primary CPT-compression method.** Fenton & Neil methodology for monotonic relationships. Noisy-OR / Noisy-MAX deprioritised because most Hormuz-style relationships are non-additive.
4. **Three-tier calibration.** Tier 1 (translator-level via golden set, Plan 1 D2). Tier 2 (intermediate-node outcome tracking, Plan 3 Layer 5). Tier 3 (Bayes-factor / regime-trajectory calibration, deferred until latent regime in production).
5. **Latent-regime topology (Step 0 — see Section 3).** $S \to D, T, P$ replaces $D, T, P \to S$. Scenarios become latent regime variables generating intermediate outcomes; the current labelling CPT is removed; a regime prior $\pi(S)$ is added. Bayes factors become first-class. This is the project's most consequential conceptual decision; the full treatment is in Section 3 (Step 0) and `notes/latent_regime_math.md`; engineering implementation lives in Plan 2 Phase 3 with cross-cutting touches in all four plans.

### UI and visualisation

1. **Refactor before polish.** Plan 4 Category A (split dashboard.py, extract CSS, fix engine caching, topological DAG levels) precedes Category C (visualisation improvements). Cleaner module boundaries make every subsequent V improvement easier.
2. **Smooth robustness gradient, retain emoji for coarse summary.** No hard category flips (no more 7.9pp 🟢 → 8.1pp 🟡); emoji stays as a quick-glance signal.
3. **CVD-safe palette.** Wong's 2011 palette (`#0072B2`, `#E69F00`, `#D55E00`) replaces the current red/green-collision-prone scheme. Plus line-style encoding for chart redundancy.
4. **Drag-to-simplex + anchor mode for sliders.** Both override patterns offered toggleable; analysts pick per task.

### Decisions deferred

The list below is the union of open questions surfaced across Plans 1–4. The "blocks" column names the earliest plan item that needs the decision resolved.


| #   | Question                                                    | Blocks                               | Notes                                                                                                                                                                                                                                        |
| --- | ----------------------------------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 18  | Per-deployment Postgres setup automation                    | Plan 3 Layer 0                       | Docker Compose for engineering MVP; Terraform / Helm for production. Choose toolchain when first deployment is concrete.                                                                                                                     |
| 19  | Auth provider (hosted vs self-hosted)                       | Plan 3 Layer 0                       | Hosted (Auth0, Clerk) for v1 speed; self-hosted (Keycloak, Authentik) considered if regulatory customers demand it.                                                                                                                          |
| 20  | LLM model for proposal generation                           | Plan 3 Layer 5                       | Coupled to Plan 1's translator provider choice; settle when Plan 1 D1/D3 are stable.                                                                                                                                                         |
| 21  | Cooke calibration question set per deployment               | Plan 3 Layer 2                       | Hormuz-specific seed questions need authoring with domain experts. Defer until Plan 3 Layer 2 ships and a first Cooke deployment is concrete.                                                                                                |
| 22  | Flex/grid DAG canvas                                        | Plan 4 C13                           | Streamlit primitives don't support cleanly. Defer until UI-framework migration.                                                                                                                                                              |
| 23  | Sampler choice for discrete latents                         | Plan 2 Phase 2                       | Default: analytic marginalization for $S$ (low cardinality, exact). NUTS + CompoundStep for larger discrete sets. Decide if needed beyond 3-state.                                                                                           |
| 24  | Per-CPT $\kappa$ values for the latent-regime emission CPTs | Plan 2 Phase 3                       | Phase 3 ships with provisional uniform $\kappa = 20$ values authored alongside the emission CPTs; per-CPT $\kappa$ becomes the elicited output of Plan 3 Layer 4. See Plan 2 Phase 3 deliverables for the provisional-then-elicited pathway. |
| 25  | Continuous oil-price data source                            | Plan 2 Phase 5                       | Bloomberg, FRED, Quandl, EIA? Decide data source and update cadence.                                                                                                                                                                         |
| 26  | Translator extension for continuous observations            | Plan 2 Phase 5 / Plan 1 A1 follow-on | Headlines like "oil hit 148" → point observations; "oil between 140-150 this week" → interval. LLM prompt extension needed.                                                                                                                  |
| 27  | UI-upgrade trigger criterion                                | Plan 3 Layer 3 / Plan 4 (future)     | When does Streamlit stop being sufficient? Define the criterion (e.g., > N concurrent users per deployment, > M custom UI components needed).                                                                                                |
| 28  | Open-source license choice                                  | Plan 3 Layer 0                       | MIT, Apache 2.0, BSD-3? Recommend Apache 2.0 for patent grant; compatible with open-core commercial layer.                                                                                                                                   |
| 29  | Stacked-bar component implementation                        | Plan 4 C8                            | Pure Altair? Custom Streamlit component? Decide before C8 starts.                                                                                                                                                                            |
| 30  | CVD-safe palette stakeholder validation                     | Plan 4 C9                            | Stakeholder review with someone CVD-affected. Optional but worth it.                                                                                                                                                                         |
| 31  | Bayes-factor display in observed-node panel                 | Plan 4 C4                            | Depends on Plan 2 Phase 3 landing. If still in pgmpy-only mode, fall back to delta display.                                                                                                                                                  |


---

**End of master plan.** Companion plans:

- `docs/translator_robustification.md` — Plan 1.
- `docs/pymc_integration_plan.md` — Plan 2.
- `docs/elicitation_tool_plan.md` — Plan 3.
- `docs/dashboard_ui_plan.md` — Plan 4.

Underlying review: `docs/dashboard_review_2026-05.md`. Pre-existing feature roadmap: `docs/bn_app_next_steps.md`. Foundational math reference (local): `notes/latent_regime_math.md`.