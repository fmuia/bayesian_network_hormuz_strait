# Master Plan: From Demo to Productisable Platform

> **Status.** Living document. M5 (material DAG omissions) remains a Gap pending a future structure-review workstream.
>
> **Purpose.** This is the orchestrator. It states the product vision, summarises the five plans (Plan 1 = latent-regime reframing; Plans 2–5 = the four engineering plans), maps every diagnosed finding to its plan, inventories all documentation in the repository, and records the strategic decisions made across the planning conversations. Start here.
>
> **Client brief.** *"Build a model that outputs the probability of each state of the Scenario node as new information comes in."* This is the founding ask. The current dashboard does not deliver it directly: it reports $E[\text{label}(D, T, P) \mid E]$, the expectation of a labelling function under the joint posterior of three intermediate-outcome parents — not a Bayesian posterior over a regime variable (finding M1). **Plan 1** is the structural fix that turns the output into $P(S \mid E)$, a genuine posterior over scenario states. **Plan 2** is the evidence-ingestion path that converts "new information" (headlines, articles) into the BN evidence $E$ that drives that posterior. Plans 1 and 2 together are the minimum-viable response to the brief; Plans 3–5 widen the engine (continuous variables, hierarchical priors), the methodology (calibrated CPT elicitation), and the dashboard surface. The "as new information comes in" clause is delivered today by Bayesian conditioning on each fresh observation (the dashboard already updates $P(S \mid E)$ when evidence is added) — note that this is **order-invariant**: two pieces of evidence arriving in either order yield the same posterior, older evidence does not decay, and the regime itself does not evolve between updates. Explicit temporal dynamics on the regime variable ($P(S_t \mid E_{1:t})$ with a state-transition matrix or HMM coupling) are the stronger reading of the brief and are out of scope for all five plans, tracked in `docs/bn_hmm_integration.md`; see [Gaps](#gaps-and-explicit-non-goals).
>
> **Status legend.** ⬜ not started · ⏳ in progress · ✅ shipped (with date).

## Table of contents

1. [Vision](#vision)
2. [Architecture overview](#architecture-overview)
3. [Plan sequence: the five plans](#plan-sequence)
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
- **Inference engine** — exact or sampling-based inference over Bayesian networks with discrete, continuous, or mixed nodes; hierarchical priors over CPTs; temporal extension for regime-tracking (longer-horizon platform capability; out of scope for Plans 1–5, see §6 Gaps).
- **Methodology layer** — multi-protocol CPT elicitation (Cooke, IDEA, SHELF), versioned artefacts with full provenance, sensitivity-driven prioritisation, multi-expert aggregation, and calibration tracking.
- **Dashboard and UI** — the analyst-facing surface that exposes the three backend layers to stakeholders, with interactive visualisation, scenario exploration, audit views, and HITL review.

**Product positioning.** Methodology-as-product with open-core licensing. The inference engine, mathematical primitives, and protocol implementations are open-source under a permissive license; the commercial layer (deployment automation, hosted versions, premium integrations, support) is closed. Deployment shape is **multi-deployment, single-tenant per engagement** — each customer engagement gets its own isolated stack rather than sharing infrastructure SaaS-style. The Hormuz implementation is the reference engagement; methodology and code reuse across engagements is the productisation story.

**Differentiated against:** academic Bayesian-network tools (AgenaRisk, Netica, Hugin, GeNIe), which are general-purpose and don't ship with elicitation methodology or calibration tracking; commercial intelligence platforms (Recorded Future, RANE, Maplecroft), which are productised but methodologically opaque; forecasting platforms (Good Judgment Inc., Hypermind), which are calibration-strong but not BN-focused.

---



## 2. Architecture overview

```
┌──────────────────────────────────────────────────────────────────────┐
│ Dashboard / UI Layer    (Plan 5 — 05_dashboard_ui_plan.md)              │
│ - Streamlit-first; modular component library                          │
│ - Backend-agnostic via the Posterior interface                        │
└──────────────────────────────────────────────────────────────────────┘
        ▲                ▲                    ▲                ▲
        │                │                    │                │
┌───────┴──────┐  ┌──────┴───────┐  ┌─────────┴──────┐  ┌──────┴─────┐
│ Evidence     │  │ Inference    │  │ Methodology    │  │ Shared     │
│ ingestion    │  │ engine       │  │ layer          │  │ substrate  │
│ (Plan 2)     │  │ (Plan 3)     │  │ (Plan 4)       │  │            │
│              │  │              │  │                │  │ audit log  │
│ translator,  │  │ pgmpy +      │  │ Cooke / IDEA / │  │ HITL queue │
│ ensemble,    │  │ PyMC dual    │  │ SHELF protocols│  │ versioning │
│ HITL review, │  │ backend with │  │ multi-expert   │  │ sources    │
│ provenance,  │  │ NetworkSpec, │  │ aggregation,   │  │ calibration│
│ RAG memory   │  │ continuous,  │  │ ranked nodes,  │  │ provenance │
│              │  │ latent regime│  │ sensitivity    │  │            │
└──────────────┘  └──────────────┘  └────────────────┘  └────────────┘
```

The arrows in the diagram represent runtime data flow:

- Plan 2 produces structured evidence; Plan 3 consumes it.
- Plan 4 produces calibrated CPTs; Plan 3 consumes them via `NetworkSpec`.
- Plan 3 produces posteriors; Plan 5 visualises them.
- Plans 2 and 4 share the audit log, HITL queue, and source-credibility registry; Plan 2 builds these, Plan 4 extends them.

The plans run **sequentially** (1 → 2 → 3 → 4 → 5) with internal items in each plan having their own ordering. Sequencing is by dependency, not just convenience. **Plan 1 is the latent-regime reframing** — both the conceptual decision (already made) and the engineering implementation that delivers it. The conceptual half underlies the specification of Plans 2–5; the engineering half ships first, editing `src/network.py`, `src/cpt_data.py`, and `src/inference.py` directly on the existing pgmpy code path (no `NetworkSpec`, no backend abstraction). Plan 3 later lifts Plan 1's work into the declarative `NetworkSpec` / dual-backend architecture and adds PyMC-native latent-regime support in its Phase 2. See §3 below for the full treatment.

---



## 3. Plan sequence: the five plans

The full work programme has five plans, executed in order **Plan 1 → Plan 2 → Plan 3 → Plan 4 → Plan 5**. **Plan 1** is the latent-regime reframing — the most consequential conceptual decision the project has made (already resolved), plus the engineering implementation that delivers it (not yet started; ships first on the existing pgmpy code path, no engineering prerequisites). **Plans 2–5** continue the engineering programme on top of Plan 1's regime topology.

### Plan 1 — Latent regime reframing: `docs/01_latent_regime_plan.md`

The current network labels scenarios via a softmax-like deterministic CPT over three intermediate-outcome nodes — `Energy_Infrastructure_Damage`, `Conflict_Duration`, `Diplomatic_Resolution_Path` (written $D$, $T$, $P$ throughout) — which finding M1 in §4 identifies as a labelling function rather than a generative probabilistic model. The reported "Stress 42%" headline is mathematically the expected value of this labelling function under the joint posterior of those parents, not a Bayesian posterior over a regime variable.

This plan applies the scenario-as-latent BN framework (`docs/scenario_bn_framework.md`) to the Hormuz network. $S$ becomes a **latent intermediate node** with parents $\text{Pa}(S) = \{M, C\}$ (the downstream-most layer of mediators) and emissions $\{D, T, P\}$ alongside their existing upstream causes. Inference becomes a genuine Bayesian posterior:

$$
P(S = s \mid E) \;\propto\; \sum_{m, c} P(M = m, C = c \mid E_{\text{up}}) \cdot P(S = s \mid m, c) \cdot P(E_{\text{down}} \mid S = s, m, c, \ldots),
$$

with Bayes factors $\Lambda_{s_1, s_2} = P(E \mid S = s_1) / P(E \mid S = s_2)$ as first-class outputs. The full derivation, the entropy-based diagnostic for M1, and the d-separation analysis live in `docs/01_latent_regime_plan.md` Section A; the underlying framework (five node categories, six edge rules, diagnostic procedure) is in `docs/scenario_bn_framework.md`.

The plan has three deliverable tracks:

- **Section A — Conceptual decision (✅ resolved).** Documents the framing in framework language, why M1 made it necessary, the math foundation, and the cross-cutting impact on Plans 2–5.
- **Section B — Engineering implementation (⬜ not started).** Edits to `src/network.py` (rewire DAG), `src/cpt_data.py` (anchor-derived emission CPTs and regime CPT $P(S \mid M, C)$, replaced by Plan 4 Layer 2 elicitation when that lands), and `src/inference.py` (three-clamped-inferences helper for Bayes-factor extraction alongside the existing `BNInferenceEngine`). No `NetworkSpec`, no backend abstraction — Plan 1 ships on the existing pgmpy code path. PyMC-native support for the latent regime is added by Plan 3 Phase 2 when `PymcBackend` lands.
- **Framework write-up (⬜ not started).** `docs/scenario_bn_framework.md` — the foundational reusable IP this plan instantiates. Can ship in parallel with engineering.

**Sequencing.** Plan 1 ships first, before Plan 2 begins. It has no engineering prerequisites — the math is implementable in pgmpy today via the existing `BNInferenceEngine`. Plan 3 later lifts Plan 1's work into the `NetworkSpec` / dual-backend architecture (Phase 0 extracts the modified network into `NetworkSpec`; Phase 1 wraps `BNInferenceEngine` and Plan 1's Bayes-factor helper inside `PgmpyBackend`; Phase 2 adds `PymcBackend` with PyMC-native latent-regime support, validated for parity against Plan 1's pgmpy implementation). Plan 3's later phases (Phase 3 continuous variables, Phase 4 Oil_Price migration) build on the regime topology — direct dependency for $\{D, T, P\}$ if they go continuous, transitive for Oil_Price (Phase 4's target, sitting in $\mathcal D$) via $P(D \mid S, M, C)$. The execution order is therefore: Plan 1 → Plan 2 → Plan 3 (Phases 0 → 1 → 2 → 3 → 4) → Plan 4 → Plan 5. The framework write-up has no engineering prerequisites and can ship at any time.

**Cross-cutting touches** in Plans 2–5 are summarised in Section C of `01_latent_regime_plan.md`. Briefly: Plan 2's likelihood-ratio output (A1) feeds Bayes-factor decomposition naturally; Plan 4 Layer 2 elicits the new emission CPTs and the regime CPT; Plan 5 C4 (rich observed-node panel) exposes Bayes-factor contributions once Plan 1's engineering lands.

This plan closes finding M1 (shipped 2026-06-05). **Finding M7 was reassessed during implementation: it is *not* closed by the reframe** — the point-estimate vs resample-mean gap is a small Jensen (non-linearity) artefact present in *both* topologies, not a consequence of the labelling CPT. See §4 and `docs/01_latent_regime_comparison.md`.

### Plan 2 — Evidence ingestion: `docs/02_translator_robustification.md`

The translator turns unstructured news into BN evidence. The current implementation is a 509-line demo; this plan converts it into a tool with calibrated uncertainty, span-grounded reasoning, multi-model cross-checks, versioned prompts, structured audit trails, and HITL review.

13 items across five categories (A–E), in 13 execution slots: B1 splits into B1a + B1b (two slots) and B4 rides inside B2's slot (no slot of its own), so item count and slot count coincide at 13.

- **Semantic foundations**: A1 (likelihood semantics fix — closes the M2 bug that mispositions every CI in the dashboard; carries an optional pairwise-Bayes-factor elicitation variant that cancels the LLM's implicit prior and produces Plan 1's $\Lambda$ directly), A2 (schema hardening).
- **Input and reasoning**: B1a (article-level input + default per-source-type credibility), B1b (per-source credibility editing with history), B2 (span-grounded structured reasoning), B3 (abstention), B4 (untrusted-input handling — spotlighting + span-grounding injection backstop + injection canary; ships with B2).
- **Uncertainty quantification**: C1 (self-consistency ensemble), C2 (multi-model cross-check).
- **Governance and evaluation**: D1 (versioned prompts), D2 (golden set + continuous evaluation, accelerated by LLM-as-judge pre-labelling and closed-loop with a post-hoc calibration map), D3 (provenance audit log).
- **Operational integration**: E1 (HITL review queue), E2 (retrieval-augmented translation).

This plan also builds the **shared substrate** (audit log, versioned-artefact pattern, HITL queue, source-credibility registry) that Plan 4 reuses: Plan 4 Layer 0 extends Plan 2's audit schema, and Plan 4 Layer 5 retrieves analyst-approved translations from the audit log to seed LLM-proposed CPT initial values. Plan 5 does not consume the substrate directly — its scope is the dashboard surface, not the persistence layer.

### Plan 3 — Inference engine: `docs/03_pymc_integration_plan.md`

The current pgmpy backend handles purely discrete networks with point-estimate CPTs and bolt-on Dirichlet resampling. This plan introduces a dual-backend architecture (`PgmpyBackend` for fast discrete inference, `PymcBackend` for hierarchical priors and continuous variables). The declarative `NetworkSpec` is the common input.

5 phases (0–4). Plan 1 has already shipped the latent regime in `src/network.py`, `src/cpt_data.py`, and `src/inference.py` before Plan 3 starts (closing M1; M7 reassessed as not closed by the reframe — see §4); Plan 3's job is to lift that work into the declarative architecture and add PyMC.

- **Phase 0**: refactor to introduce `NetworkSpec` (lifting the post-Plan-1 `src/network.py` into the declarative form).
- **Phase 1**: wrap existing inference (`BNInferenceEngine`, `sensitivity.py`, and Plan 1's Bayes-factor helper) in `PgmpyBackend` with the uniform `Posterior` interface.
- **Phase 2**: build `PymcBackend` for discrete-only networks, including PyMC-native latent-regime support; validate parity against Plan 1's pgmpy implementation.
- **Phase 3**: continuous variable support (closes M2 properly at the inference layer, plus M3 via per-CPT κ and M4 via hierarchical priors).
- **Phase 4**: migrate Oil_Price to continuous in production.

Temporal extensions to the BN (a Markov chain on the regime variable) and BN↔HMM integration are **out of scope** for this plan. They require a separately-trained inflation HMM that does not exist in this repository and are tracked in `docs/bn_hmm_integration.md` as a longer-horizon workstream; see [Gaps](#gaps-and-explicit-non-goals). Roadmap A1 (evidence accumulation) is therefore not delivered by any plan.

Mathematical context for the migration lives in `docs/01_latent_regime_plan.md` Section A.

### Plan 4 — Methodology layer: `docs/04_elicitation_tool_plan.md`

CPTs in the current model are inline literals chosen by one author without protocol. This plan delivers a multi-protocol elicitation platform: Cooke's classical model for regulatory contexts, IDEA for corporate decisions, SHELF for solo analysts. Includes versioned CPTs with full provenance, multi-expert aggregation, ranked-node compression, sensitivity-driven prioritisation, three-tier calibration tracking, and LLM-proposed CPT initial values via retrieval from Plan 2's audit log.

Layers 0–5 in scope (6 layers); Layer 6 (commercial: billing, onboarding, support) deferred until paying customers exist:

- **Layer 0**: data model and storage substrate; extends Plan 2's audit schema.
- **Layer 1**: core engine (aggregation primitives, ranked nodes, sensitivity analysis).
- **Layer 2**: protocol implementations.
- **Layer 3**: Streamlit UI.
- **Layer 4**: integration with Plan 3's inference engine.
- **Layer 5**: advanced features (LLM-proposed CPTs, calibration tiers 2 and 3, Cooke weight updates from accrued outcomes).

### Plan 5 — Dashboard UI and polish: `docs/05_dashboard_ui_plan.md`

Closes the visualisation, dashboard-architecture, performance, code-hygiene, and test-coverage findings that the three backend plans leave open.

5 categories with 29 items:

- **Category A — architectural refactor**: split `dashboard.py`, extract CSS, fix engine caching, derive DAG levels topologically.
- **Category B — performance**: bound caches, memoise evolution series, apply-button override pattern.
- **Category C — visualisation improvements**: 14 items. C1–C13 address visualisation findings V2–V15 (V8 is handled by Category A's CSS extraction; V1 is covered by Plan 1's latent-regime work). C14 owns the continuous Oil_Price panel and interval queries that Plan 3 Phase 3/4 enables — the work is in Plan 5 rather than Plan 3 so dashboard ownership stays in one place.
- **Category D — code hygiene**: dead-code triage, type-vocabulary cleanup, defensive-guard documentation, deduplication.
- **Category E — test coverage**: tests for state helpers, visualisation primitives, and dashboard components.

Several items in Categories A, D, and E are backend-independent and can be picked up opportunistically alongside the other plans.

---



## 4. Findings coverage matrix

This matrix is the canonical, in-tree registry of every M/C/V finding (originating from a local dashboard-review pass) and the plan that closes it. The Title column is the load-bearing description; if you want the full context for a finding, the plan referenced in the Plan column is the place to look. If a finding is not in this matrix, it is not covered anywhere — please raise it.

### Math findings (M*)


| ID  | Title                             | Plan                                                | Section                                                                               |
| --- | --------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------- |
| M1  | Scenario as classifier            | Plan 1                                              | Section A (decision) + Section B (engineering implementation)                         |
| M2  | Soft evidence semantics           | Plan 2 + Plan 3                                     | A1 + Phase 3                                                                          |
| M3  | Uniform κ                         | Plan 3 + Plan 4                                     | Phase 3 (per-CPT κ in `NetworkSpec`) + Layer 4 (per-CPT κ via elicitation provenance) |
| M4  | Independent CPT column resampling | Plan 3                                              | Phase 3 (hierarchical priors)                                                         |
| M5  | Material DAG omissions            | **Not in any plan**                                 | See [Gaps](#gaps-and-explicit-non-goals) below                                        |
| M6  | Root priors unjustified           | Plan 4                                              | Layer 2 (priors elicited via Cooke or IDEA)                                           |
| M7  | Resample-mean vs point-estimate   | Plan 1 (reassessed — NOT closed)                    | Empirically the point-vs-resample gap is a small Jensen artefact (~≤1.3pp, 95th-pctile <1pp) present in *both* topologies, not caused by the labelling CPT. The reframe does not shrink it; the dashboard already reports the resample-mean. See `docs/01_latent_regime_comparison.md` (Finding 3). |
| M8  | `+1e-6` Dirichlet guard           | Plan 5                                              | D3                                                                                    |
| M9  | Duplicate sensitivity functions   | Plan 5                                              | D4                                                                                    |


### Code findings (C*)


| ID  | Title                                       | Plan            | Section      |
| --- | ------------------------------------------- | --------------- | ------------ |
| C1  | `engine` cached but mutated                 | Plan 5          | A3           |
| C2  | `dashboard.py` 1878 lines                   | Plan 5          | A1           |
| C3  | Probability evolution recomputes everything | Plan 5          | B2 + B3      |
| C4  | Caches unbounded                            | Plan 5          | B1           |
| C5  | Soft-evidence mismatch (code side of M2)    | Plan 2 + Plan 3 | A1 + Phase 3 |
| C6  | Translator validation permissive            | Plan 2          | A2           |
| C7  | Silent renormalization                      | Plan 2          | A2           |
| C8  | Regex JSON extraction fragile               | Plan 2          | A2           |
| C9  | `render_network_png` dead code              | Plan 5          | D1           |
| C10 | `_NODE_LEVEL` hardcoded                     | Plan 5          | A4           |
| C11 | No `_merged_evidence` test                  | Plan 5          | E1           |
| C12 | `_PLUGINS_REGISTERED` workaround            | Plan 5          | D5           |
| C13 | `Observation.tone` unused                   | Plan 5          | D2           |
| C14 | No `viz.py` / dashboard helper tests        | Plan 5          | E2 + E3      |


### Visualisation findings (V*)


| ID  | Title                                   | Plan             | Section                                                 |
| --- | --------------------------------------- | ---------------- | ------------------------------------------------------- |
| V1  | Headline = resample mean                | Plan 1           | The dashboard already plots the resample-mean (correct centre of the CI band); the residual point-vs-resample gap (M7) is a small Jensen artefact, not removed by the reframe. |
| V2  | Info density wrong                      | Plan 5           | C1                                                      |
| V3  | Robustness thresholds arbitrary         | Plan 5           | C2                                                      |
| V4  | Slider UX trap                          | Plan 5           | C3                                                      |
| V5  | Hard-observed uninformative bars        | Plan 5           | C4                                                      |
| V6  | DAG layout crowded                      | Plan 5           | C5                                                      |
| V7  | Param vs forecast uncertainty conflated | Plan 5           | C6                                                      |
| V8  | CSS inline                              | Plan 5           | A2                                                      |
| V9  | No before/after on new observation      | Plan 5 + roadmap | C7 (delta-chip) + roadmap A3 (full waterfall)           |
| V10 | Latest-translation as plain text        | Plan 5           | C8                                                      |
| V11 | Colourblind-hostile palette             | Plan 5           | C9                                                      |
| V12 | Tooltip headline truncation             | Plan 5           | C10                                                     |
| V13 | Edge rationale isolated                 | Plan 5           | C11                                                     |
| V14 | Fixed narrative paragraphs              | Plan 5 + roadmap | C12 (hide for now) + roadmap B1 (responsive narratives) |
| V15 | DAG canvas height brittle               | Plan 5           | C13 (likely deferred)                                   |


### Summary

- **38 findings tracked** in this matrix (9 M-findings, 14 C-findings, 15 V-findings).
- **37 findings have a home** in one of the five plans.
- **1 finding (M5)** is explicitly not covered; see [Gaps](#gaps-and-explicit-non-goals).

---



## 5. Existing roadmap items mapping (`docs/bn_app_next_steps.md`)

The pre-existing feature roadmap contains 11 items (A1–E1). Their relationship to the five plans:


| Roadmap item | Title                         | Status / Where it lives                                                                                                                                                                                      |
| ------------ | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A1           | Evidence accumulation         | **Not in any plan.** The proper mechanism is a temporal Markov chain on the regime (or coupling to a separately-trained inflation HMM). Both require infrastructure that does not exist in this repository; see [Gaps](#gaps-and-explicit-non-goals) and `docs/bn_hmm_integration.md`. |
| A2           | Node-level credible intervals | ✅ Shipped 2026-04. Pre-dates this planning exercise.                                                                                                                                                         |
| A3           | Sensitivity attribution       | **Not in any plan.** Compounds with Plan 5 C7 (delta chip) and Plan 1 (Bayes-factor decomposition makes this natural). Could be promoted to a Plan 5 item or kept as a roadmap item for later. |
| B1           | Daily narrative generation    | **Not in any plan.** Compounds with Plan 5 C12 (responsive narratives). Requires LLM summary infrastructure beyond the translator.                                                                     |
| B2           | Pre-built scenario sequences  | **Not in any plan.** Demo-time UX; trivial to add as a follow-on to Plan 5.                                                                                                                            |
| B3           | Session export                | **Not in any plan.** `render_network_png` in Plan 5 D1 is annotated for B3's eventual use.                                                                                                             |
| C1           | Scenario comparison mode      | **Not in any plan.** UI feature; compounds with Plan 5 information architecture.                                                                                                                       |
| C2           | CPT explorer                  | Delivered by Plan 4 (the elicitation tool is the CPT explorer, generalised).                                                                                                                                 |
| C3           | Undo/redo and pinning         | **Not in any plan.** UI feature; would compound with Plan 5 state management.                                                                                                                          |
| D1           | Batch processing              | Delivered by Plan 2's article-level input and the multi-headline ingest path.                                                                                                                                |
| E1           | News memory database          | Delivered by Plan 2's E2 (RAG memory). The "two consumers, one memory" pattern (translator + narrative layer) is preserved.                                                                                  |


Roadmap items not delivered by any plan (A3, B1, B2, B3, C1, C3) are kept in `docs/bn_app_next_steps.md` as a follow-on backlog. They are largely UI-flavoured features that compound naturally with Plan 5 work but aren't structural enough to warrant dedicated plan items.

---



## 6. Gaps and explicit non-goals

### Gaps acknowledged

> **Note on the client brief.** The brief — *"output $P(S = s)$ as new information comes in"* — is delivered v1 via Bayesian conditioning on each new piece of evidence: Plan 1 makes $P(S \mid E)$ a genuine posterior, and Plan 2 turns each new article into the evidence $E$ that drives the update. **Explicit temporal dynamics** — $P(S_t \mid E_{1:t})$ with a state-transition matrix $P(S_t \mid S_{t-1})$ or coupling to a separately-trained inflation HMM — are a stronger reading of "as new information comes in" and are **not** delivered by any of the five plans. The temporal extension is the precursor capability the BN needs before any HMM coupling is possible; it is tracked in the second bullet below and in `docs/bn_hmm_integration.md`. Stakeholder communication on this point: today, two pieces of evidence arriving in either order yield the same posterior (Bayesian conditioning is order-invariant). A temporal model is what allows older evidence to *decay* and the regime itself to *evolve* between updates.

- **M5 — Material DAG omissions.** The review identifies real causal channels not in the current DAG, most notably `Iran_Aligned_Militia_Attacks → US_Military_Response` (the Iraq/Syria base-strike pattern). This is a **modelling decision**, not an engineering one. It belongs in a future BN-structure review with domain experts, ideally using the elicitation tool from Plan 4 once it ships. Until then, the current DAG's scoping is documented in the edge-rationale tab and the `_EDGE_OMISSIONS` list.
- **Temporal BN extension and BN↔HMM integration (entirety of `docs/bn_hmm_integration.md`).** The integration doc describes four channels for coupling a BN to a data-trained inflation HMM: structural priors (Section 3.1.5), transition-matrix covariates (Approach A, Section 3.1), hierarchical sub-states (Approach B, Section 3.2), and emission modification (Approach C, Section 3.3). All four require a separately-trained inflation HMM that does not exist in this repository. A BN-internal temporal extension (a Markov chain on the regime variable) would be a precursor capability the BN needs before any of those couplings is possible; it too is out of scope for all five plans. Roadmap A1 (evidence accumulation) is the closest in-repo touchpoint and is therefore also not delivered. The whole workstream is tracked in `docs/bn_hmm_integration.md` for the longer horizon.

### Explicit non-goals

The following are **deliberately out of scope** for all five plans:

- **A full BN-structure re-elicitation.** The five plans take the current network's edge structure as given (modulo the arrow inversion around `Scenario` that Plan 1 delivers). DAG-structure decisions are a separate workstream that has not yet been planned; see the M5 entry under "Gaps acknowledged" above.
- **Multi-tenant SaaS infrastructure.** The platform is multi-deployment, single-tenant per engagement. No `tenant_id` columns, no row-level security. If a customer eventually demands SaaS, this is a retrofit, not a v1 concern.
- **Real-time news ingestion at scale.** The translator handles demo-cadence input (≤10 articles/day). High-throughput ingest is deferred until a customer demands it.
- **Mobile-first responsive design.** Stakeholder-facing committee tool; desktop-first is correct.
- **Internationalisation.** English-only for v1; the translator and elicitation prompts assume English source material.
- **Authentication and authorisation hardening beyond per-deployment basics.** SSO integration is in scope (Plan 4 Layer 0); fine-grained per-resource RBAC is not.

---



## 7. Repository documentation inventory

Every doc in this repository, in one place.

### Top-level

- `README.md` — quick-start for running the app and the BN-vs-HMM rationale. Touchpoints per plan: Plan 2 B1 updates the news-ingestion description (article-level input + source-credibility) and E2 adds the news-memory description; Plan 1 invalidates the current model-description section (latent-regime topology replaces labelling); Plan 3 Phase 4 adds the continuous Oil_Price section; Plan 4 Layer 3 adds an elicitation-tool quick-start; Plan 5 A1 updates any code-layout description. Reviewer should sweep the README at the end of each plan.

### `docs/`


| Doc                             | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `master_plan.md` (this doc)     | Orchestrator. Vision, sequencing, coverage matrix (the in-tree registry of every M/C/V finding), decision log. Start here.                                                                                                                                                                                                                                                                                                             |
| `01_latent_regime_plan.md`      | **Plan 1.** Latent-regime reframing (Hormuz instance of the scenario-as-latent BN framework). Section A is the conceptual decision (resolved); Section B is the engineering implementation (**shipped 2026-06-05** on the existing pgmpy code path — `src/network.py`, `src/cpt_data.py`, `src/inference.py`, `src/sensitivity.py`; both topologies kept side-by-side, labelling default). Closes M1; M7 reassessed as not closed by the reframe. Comparison: `docs/01_latent_regime_comparison.md`, `notebooks/latent_regime_comparison.ipynb`. |
| `scenario_bn_framework.md`      | Foundational design pattern for scenario-as-latent BNs. Five node categories, six edge rules, diagnostic procedure. Companion to Plan 1; reusable across future scenario-BN engagements.                                                                                                                                                                                                                                                                                              |
| `02_translator_robustification.md` | **Plan 2.** Evidence-ingestion layer. 13 items, A1–E2 (incl. B4 untrusted-input handling), in 13 execution slots.                                                                                                                                                                                                                                                                                                                                                                                 |
| `03_pymc_integration_plan.md`      | **Plan 3.** Inference engine. 5 phases, 0–4. Lifts Plan 1's directly-edited network into the `NetworkSpec` / dual-backend architecture and adds PyMC-native latent-regime support in Phase 2; Phases 3–4 add continuous variables and the Oil_Price migration.                                                                                                                                                                                                                                                                                                                                              |
| `04_elicitation_tool_plan.md`      | **Plan 4.** Methodology layer. 6 active layers, 0–5 (Layer 6 deferred).                                                                                                                                                                                                                                                                                                                                                                |
| `05_dashboard_ui_plan.md`          | **Plan 5.** Dashboard UI and polish. 5 categories, 29 items.                                                                                                                                                                                                                                                                                                                                                                           |
| `bn_app_next_steps.md`          | Pre-existing feature roadmap. A1–E1 items. Several items are delivered by Plans 2–5; remaining items (A3, B1, B2, B3, C1, C3) are the follow-on backlog.                                                                                                                                                                                                                                                                               |
| `bn_hmm_integration.md`         | Longer-horizon BN↔HMM integration story (four channels: structural priors at Section 3.1.5; Approach A transition-matrix covariates at Section 3.1; Approach B hierarchical sub-states at Section 3.2; Approach C emission modification at Section 3.3). **Entirely out of scope for all five plans** — every channel needs a separately-trained inflation HMM that does not exist in this repository. Even a BN-internal temporal extension (a Markov chain on the regime variable, precursor to any of these) is out of scope. See [Gaps](#gaps-and-explicit-non-goals). |
| `model_documentation.md`        | Reference documentation for the current model: nodes, CPTs, inference mechanics.                                                                                                                                                                                                                                                                                                                                                       |


---



## 8. Decision log

Strategic decisions resolved across the planning conversations, ordered roughly chronologically.

### Product strategy

1. **Position — Methodology-as-product (Option B), consulting-first evolving toward productised on-premise (Pattern 1).** The platform is sold as both implementation services and a licensable product. Each customer engagement is a bespoke deployment; the platform is the underlying tool that codifies the methodology.
2. **Licensing — Open core.** Engine, mathematical primitives, and protocol implementations are open-source under a permissive license (Apache 2.0 recommended). Commercial layer (deployment automation, hosted versions, premium integrations, support) is closed.
3. **Deployment shape — Multi-deployment, single-tenant per engagement.** No SaaS-style multi-tenancy. Each customer gets an isolated stack with its own database, app instance, users, and configuration.
4. **Initial deployment patterns — On-premise (Pattern 1) and managed single-tenant (Pattern 2).** Customer-hosted with support contracts, or hosted by us in the customer's cloud. Avoid pure SaaS for high-stakes regulatory use cases.

### Architecture

1. **Dual-backend with capability-based dispatch.** A declarative `NetworkSpec` describes the model; a dispatcher routes to `PgmpyBackend` (fast exact for discrete) or `PymcBackend` (hierarchical priors, continuous variables, PyMC-native latent regime). Dashboard is backend-agnostic via a uniform `Posterior` interface.
2. **PyMC is mandatory when continuous variables are present.** All-discrete networks let the user choose; mixed networks must use PyMC.
3. **Shared substrate across Plans 2 and 4.** The audit log, versioned-artefact pattern, HITL queue, and source-credibility registry are built once in Plan 2 and extended in Plan 4. No duplication.
4. **Streamlit-first UI.** Streamlit forms and pages for v1. Upgrade to a dedicated web frontend (React or similar) when scaling demand or feature complexity justifies the investment.

### Methodology

1. **Likelihood semantics for translator output (Plan 2 A1).** The LLM produces likelihood ratios `ε_s = P(article | state=s) / max_s'`, not posterior-shaped distributions. Closes the translator-interface facet of M2 and its code side C5; the inference-layer facet of M2 (soft evidence on continuous nodes) is closed separately by Plan 3 Phase 3, per the §4 matrix. An optional **pairwise-Bayes-factor elicitation variant** (Plan 2 A1, deferred) asks the LLM directly for ratios `Λ_ij = P(article|s_i)/P(article|s_j)` so its implicit prior cancels; this yields Plan 1's `Λ` object directly and is gated on Plan 2 D2 calibration showing per-state prior leakage is material.
2. **Three elicitation protocols supported (Plan 4 Layer 2).** Cooke's classical model for regulatory contexts (Hormuz-style); IDEA for corporate decisions; SHELF for solo analysts. Configurable at elicitation start.
3. **Ranked nodes as the primary CPT-compression method.** Fenton & Neil methodology for monotonic relationships. Noisy-OR / Noisy-MAX deprioritised because most Hormuz-style relationships are non-additive.
4. **Three-tier calibration.** Tier 1 (translator-level via golden set, Plan 2 D2). Tier 2 (intermediate-node outcome tracking, Plan 4 Layer 5). Tier 3 (Bayes-factor / regime-trajectory calibration, deferred until latent regime in production).
5. **Latent-regime topology (Plan 1).** $S \to D, T, P$ replaces $D, T, P \to S$; $S$ gains upstream parents $\text{Pa}(S) = \{M, C\}$ (the downstream-most mediator layer). Scenarios become latent regime variables generating intermediate outcomes; the current labelling CPT is removed; a regime CPT $P(S \mid M, C)$ is added. Bayes factors become first-class. This is the project's most consequential conceptual decision; the full treatment lives in `docs/01_latent_regime_plan.md` (Section A is the decision, including the underlying math; Section B is the engineering implementation; the framework that the plan instantiates is in `docs/scenario_bn_framework.md`). Plan 1 ships first, editing `src/network.py`, `src/cpt_data.py`, and `src/inference.py` directly on the existing pgmpy code path; no `NetworkSpec` or backend abstraction is required. Plan 3 later lifts the work into the `NetworkSpec` / dual-backend architecture (Phase 0 → Phase 2), with PyMC-native latent-regime support added in Phase 2. The implementation has cross-cutting touches in Plans 2, 4, and 5 (summarised in Plan 1 Section C).
6. **Plan 1 ownership.** The conceptual decision and the engineering implementation are tracked together in `docs/01_latent_regime_plan.md`; the foundational framework write-up lives in the companion file `docs/scenario_bn_framework.md`. M5 (material DAG omissions) remains a Gap pending a future structure-review workstream.

### UI and visualisation

1. **Refactor before polish.** Plan 5 Category A (split dashboard.py, extract CSS, fix engine caching, topological DAG levels) precedes Category C (visualisation improvements). Cleaner module boundaries make every subsequent V improvement easier.
2. **Smooth robustness gradient, retain emoji for coarse summary.** No hard category flips (no more 7.9pp 🟢 → 8.1pp 🟡); emoji stays as a quick-glance signal.
3. **CVD-safe palette.** Wong's 2011 palette (`#0072B2`, `#E69F00`, `#D55E00`) replaces the current red/green-collision-prone scheme. Plus line-style encoding for chart redundancy.
4. **Drag-to-simplex + anchor mode for sliders.** Both override patterns offered toggleable; analysts pick per task.

### Decisions deferred

The list below is the union of open questions surfaced across Plans 2–5. The "blocks" column names the earliest plan item that needs the decision resolved.


| #   | Question                                                    | Blocks                               | Notes                                                                                                                                                                                                                                        |
| --- | ----------------------------------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 18  | Per-deployment Postgres setup automation                    | Plan 4 Layer 0                       | Docker Compose for engineering MVP; Terraform / Helm for production. Choose toolchain when first deployment is concrete.                                                                                                                     |
| 19  | Auth provider (hosted vs self-hosted)                       | Plan 4 Layer 0                       | Hosted (Auth0, Clerk) for v1 speed; self-hosted (Keycloak, Authentik) considered if regulatory customers demand it.                                                                                                                          |
| 20  | LLM model for proposal generation                           | Plan 4 Layer 5                       | Coupled to Plan 2's translator provider choice; settle when Plan 2 D1/D3 are stable.                                                                                                                                                         |
| 21  | Cooke calibration question set per deployment               | Plan 4 Layer 2                       | Hormuz-specific seed questions need authoring with domain experts. Defer until Plan 4 Layer 2 ships and a first Cooke deployment is concrete.                                                                                                |
| 22  | Flex/grid DAG canvas                                        | Plan 5 C13                           | Streamlit primitives don't support cleanly. Defer until UI-framework migration.                                                                                                                                                              |
| 23  | Sampler choice for discrete latents                         | Plan 3 Phase 2                       | Default: analytic marginalization for $S$ (low cardinality, exact). NUTS + CompoundStep for larger discrete sets. Decide if needed beyond 3-state.                                                                                           |
| 24  | Per-CPT $\kappa$ values for the latent-regime emission CPTs | Plan 1                               | Plan 1 ships with provisional uniform $\kappa = 10$ values on every emission CPT and on the regime CPT $P(S \mid M, C)$ (justified in Plan 1 Section B.1); per-CPT $\kappa$ becomes the elicited output of Plan 4 Layer 4. See Plan 1 Section B deliverables for the provisional-then-elicited pathway. Note: $\kappa$ is consumed by `PymcBackend` (hierarchical priors); pgmpy direct VE uses the point-estimate CPT, so Plan 1 (pgmpy-only by design) is insensitive to the $\kappa$ choice — $\kappa$ first becomes operative when `PymcBackend` lands in Plan 3 Phase 2. |
| 25  | Continuous oil-price data source                            | Plan 3 Phase 4                       | Bloomberg, FRED, Quandl, EIA? Decide data source and update cadence.                                                                                                                                                                         |
| 26  | Translator extension for continuous observations            | Plan 3 Phase 4 / Plan 2 A1 follow-on | Headlines like "oil hit 148" → point observations; "oil between 140-150 this week" → interval. LLM prompt extension needed.                                                                                                                  |
| 27  | UI-upgrade trigger criterion                                | Plan 4 Layer 3 / Plan 5 (future)     | When does Streamlit stop being sufficient? Define the criterion (e.g., > N concurrent users per deployment, > M custom UI components needed).                                                                                                |
| 28  | Open-source license choice                                  | Plan 4 Layer 0                       | MIT, Apache 2.0, BSD-3? Recommend Apache 2.0 for patent grant; compatible with open-core commercial layer.                                                                                                                                   |
| 29  | Stacked-bar component implementation                        | Plan 5 C8                            | Pure Altair? Custom Streamlit component? Decide before C8 starts.                                                                                                                                                                            |
| 30  | CVD-safe palette stakeholder validation                     | Plan 5 C9                            | Stakeholder review with someone CVD-affected. Optional but worth it.                                                                                                                                                                         |
| 31  | Bayes-factor display in observed-node panel                 | Plan 5 C4                            | Available from Plan 1 onward via the three-clamped-inferences helper on pgmpy direct VE. `PymcBackend` exposes the same `Posterior.bayes_factor` API via samples in Plan 3 Phase 2. No fallback to delta display is required once Plan 1 has shipped. |


---

**End of master plan.** Companion plans:

- `docs/01_latent_regime_plan.md` — Plan 1.
- `docs/02_translator_robustification.md` — Plan 2.
- `docs/03_pymc_integration_plan.md` — Plan 3.
- `docs/04_elicitation_tool_plan.md` — Plan 4.
- `docs/05_dashboard_ui_plan.md` — Plan 5.

Pre-existing feature roadmap: `docs/bn_app_next_steps.md`.