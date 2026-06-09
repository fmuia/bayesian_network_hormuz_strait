# Elicitation Tool Plan: A Cooke-Protocol CPT Elicitation Platform

> **Companion document.** The methodology, design rationale, defensibility argument, and full reference list live in [`docs/elicitation_methodology_and_defensibility.md`](elicitation_methodology_and_defensibility.md). This document is the **executable plan**: layers, deliverables, file paths, and validation criteria. Where a design choice needs justification, this plan states the choice and links to the methodology document for the argument.
>
> **Status legend.** ⬜ not started · ⏳ in progress · ✅ shipped (with date). All layers are ⬜.

## Executive summary

This plan delivers a tool for **eliciting** the conditional probability tables (CPTs) that drive the Bayesian network, **aggregating** the independent judgments of several experts into a single calibrated distribution, **versioning** every CPT with full provenance, **propagating** per-CPT uncertainty into the model's outputs, and **tracking calibration** of those CPTs against realised outcomes.

The elicitation protocol is **Cooke's classical model** — seed-validated, performance-weighted, with poorly-calibrated experts zeroed. Two further structured-expert-judgment protocols, **IDEA** and **SHELF**, are documented in the [methodology companion](elicitation_methodology_and_defensibility.md) §3 and kept behind the same `Protocol` / `Expert` interface so they can be added later without restructuring — but they are **out of implementation scope**. Nothing in this plan builds them.

The platform addresses the deepest epistemic weakness of the current model: CPT values are inline literals chosen by one author, with no formal protocol, no multi-expert input, and no calibration tracking. The current dashboard already *propagates* CPT uncertainty (a Dirichlet resampling pass in [`src/sensitivity.py`](../src/sensitivity.py) that the dashboard renders as 80% credible intervals), but with a single hard-coded concentration `κ = 20` applied uniformly to every CPT and around one-author point values. This plan replaces both: point values come from a scored panel, and each CPT carries its own calibration-derived `κ`.

**Operating mode: AI-only expert panels, human-capable by construction.** The expert *panel members* are multi-model AI agents. Because the panel is AI-only, the AI-expert capability is **on the critical path** — it is part of the elicitation core (Layer 2), not a final add-on. Human experts remain a first-class, retained option: an `LLMExpert` and a human expert implement the **identical** `Expert` interface, so enabling human panellists is a per-deployment configuration choice, not a code change. "AI-only" refers to the panel members, not the reviewer — **human sign-off remains mandatory for high-stakes work.**

The platform is built in **seven layers, foundation first**:

| Layer | Scope | Milestone |
| --- | --- | --- |
| **0** | Data model and storage substrate (extends the translator audit log) | |
| **1** | Core engine — Cooke aggregation, sensitivity, the calibration→κ mapping | |
| **2** | Cooke elicitation + AI experts — state machine, `Expert` interface, single-agent `LLMExpert`, seed scoring | *it runs* |
| **3** | Multi-model panels & defensibility hardening — decorrelation, contamination probes, calibration reports, red-team | *it's defensible* |
| **4** | Streamlit UI | **v1 (Layers 0–4)** |
| **5** | Inference integration — `NetworkSpec`, per-CPT κ into the engine and dashboard | |
| **6** | Calibration tracking & confidence over time — Tier 2/3, weight/κ updates, prioritisation, confidence report | |

**Why this order.** Layers 0–2 are the minimal runnable elicitation pipeline: storage, the maths, and a Cooke workflow an AI agent can actually drive. Layer 3 makes that pipeline *defensible* (multi-model diversity and contamination probes are not optional polish — single-model naive pooling is the failure mode the methodology warns against, so hardening comes immediately after the core, not last). Layer 4 (UI) completes the **v1 milestone**: a usable, defensible, AI-only Cooke elicitation tool. Layers 5–6 connect the elicited CPTs back to the live model and accrue calibration over time.

**Deployment shape: multi-deployment, single-tenant per deployment.** Each deployment runs in its own isolated stack — own database, own users, own configuration — with no data co-mingling. A data-governance requirement driven by the sensitivity of the source material, not a shared multi-tenant database.

## Context

### Position in the plan stack

This plan has two dependencies and one ownership responsibility:

- **Plan 1 — [`docs/01_latent_regime_plan.md`](01_latent_regime_plan.md) (topology).** Determines which CPTs exist, and therefore what Layer 2 elicits. Under the latent-regime topology the protocol elicits emission CPTs ($P(D\mid S)$, $P(T\mid S)$, $P(P\mid S)$) and a regime prior $\pi(S)$ rather than a labelling CPT. The questions become generative ("given the regime, what do the observables look like?"), which is easier to defend with domain experts. The protocol is topology-agnostic; the active topology is named in deployment configuration.
- **Plan 2 — [`docs/02_translator_robustification.md`](02_translator_robustification.md) (substrate).** Provides the audit-log schema, versioned-artefact pattern, HITL review queue, source-credibility registry, and golden-set evaluation harness — all reused here rather than duplicated.
- **`NetworkSpec` ownership.** This plan ships the **discrete** subset of the declarative `NetworkSpec` (`DiscreteNode`, `NetworkSpec`) in `src/network_spec.py` and integrates with the existing pgmpy engine ([`src/inference.py`](../src/inference.py)). The continuous-node and PyMC-backend extension is the subject of [`docs/03_pymc_integration_plan.md`](03_pymc_integration_plan.md); when that lands it extends the same file. The export interface defined here is stable across that change.

### Position relative to existing tools

Examined against existing tools (each verified against its vendor or primary source):

- **[AgenaRisk](https://www.agenarisk.com/), [Netica](https://www.norsys.com/netica.html), [Hugin](https://www.hugin.com/), [GeNIe](https://www.bayesfusion.com/genie/).** General-purpose Bayesian-network tools with CPT-editor UIs and CPT-compression aids. Strong on mechanical CPT-editing; weak on elicitation methodology, multi-expert aggregation, and calibration tracking. We borrow the visual CPT-editor pattern and replace the spreadsheet-style elicitation UI with protocol-driven workflows.
- **[Good Judgment Inc.](https://goodjudgment.com/), [Metaculus](https://www.metaculus.com/).** Forecaster ensembles with calibration tracking. Excellent on calibration; not designed for structured CPT elicitation. We borrow the calibration-tracking pattern (performance weights from realised accuracy) and apply it to per-expert CPT contributions.
- **Catastrophe modelling ([Moody's RMS](https://www.rms.com/), [Verisk](https://www.verisk.com/insurance/products/extreme-event-solutions/)).** Mature expert elicitation for tail risk; tooling proprietary. We borrow the anchored-elicitation-against-analogs pattern and sensitivity-driven prioritisation.
- **Regulatory methodology ([SHELF](https://shelf.sites.sheffield.ac.uk/), [EFSA](https://efsa.onlinelibrary.wiley.com/doi/10.2903/j.efsa.2014.3734), [IPCC](https://www.ipcc.ch/site/assets/uploads/2018/05/uncertainty-guidance-note.pdf)).** Open methodology documents on defensible elicitation. We implement Cooke directly and adopt the IPCC two-dimensional uncertainty language for reporting.
- **Geopolitical intel platforms ([Recorded Future](https://www.recordedfuture.com/), [RANE](https://www.ranenetwork.com/)).** Operationally polished, methodologically opaque. Our approach is the inverse: methodologically transparent and auditable.

The gap this work occupies: **methodologically rigorous, calibration-aware CPT elicitation — extended to calibration-validated AI expert panels — for high-stakes analytical contexts.** No off-the-shelf tool covers this space.

## Diagnosis: why the current state is insufficient

This is the failure surface the plan closes. It maps to two master-plan findings — **M6** (root priors unjustified, closed by Layer 2) and **M3** (uniform κ, closed by Layers 1 and 5) — plus roadmap item **C2**.

1. **CPT values are inline literals chosen by one author without protocol.** [`src/network.py`](../src/network.py) contains hand-tuned probabilities with brief comments as justification. No record of who picked the numbers, when, against what reference, or with what confidence.
2. **No multi-expert aggregation.** No infrastructure to elicit several views independently and combine them.
3. **No calibration tracking.** No record of which CPT entries produced predictions that matched outcomes.
4. **No CPT versioning.** Previous values are lost to git history at best.
5. **No per-CPT provenance.** Every cell has the same epistemic status: "the author chose this."
6. **Uncertainty is propagated but not calibrated.** [`src/sensitivity.py`](../src/sensitivity.py) already resamples every CPT from a Dirichlet and reports credible intervals — but with a single global `κ = 20` and one-author means. The mechanism is right; the inputs are placeholders, so the interval widths carry no calibrated meaning.
7. **No sensitivity-driven prioritisation.** Effort is distributed uniformly, but a small subset of CPT entries drives most output variation.
8. **No formal elicitation protocol.** The current implementation supports none.
9. **No reuse across deployments.** Each new problem requires copy-paste-and-edit of `network.py`.
10. **No coupling to the translator's evidence corpus.** Plan 2 builds an audit log of every article translated, with span-grounded claims — exactly the analog-event database anchored elicitation needs, currently unusable for elicitation.

## Architecture

### Design principles

1. **Multi-deployment, single-tenant per deployment.** Each deployment gets its own isolated stack. No `tenant_id` columns; no row-level security; no data co-mingling.
2. **Layered, foundation first.** Each layer is independently testable and deliverable. Lower layers do not depend on higher ones.
3. **Shared infrastructure with Plan 2.** The audit log schema, versioned-artefact pattern, HITL queue, source-credibility registry, and Streamlit shell are extended, not duplicated.
4. **Owns the discrete `NetworkSpec` contract.** Elicited CPTs export to a declarative `NetworkSpec` carrying a per-CPT `κ`. The interface is stable across the later PyMC extension.
5. **Streamlit-first UI.** Streamlit forms for v1; a dedicated frontend only when scaling or feature demand requires it.
6. **AI-only panels, human-capable interface.** Panel members are AI agents today; the `Expert` interface is built human-capable so the option never has to be retrofitted.
7. **Protocol-general core, Cooke-only build.** The `Protocol`/`Expert`/`ElicitationTarget` abstractions are protocol-agnostic; only Cooke is implemented. IDEA and SHELF are documented and slot in behind the same interface later.

### Layer structure

```
┌──────────────────────────────────────────────────────────────────┐
│ Layer 6 — Calibration tracking & confidence over time             │
│  Tier 2/3 outcomes · weight & κ updates · prioritisation          │
│  · assembled confidence report · (optional) LLM-drafted CPTs      │
├──────────────────────────────────────────────────────────────────┤
│ Layer 5 — Inference integration                                   │
│  Elicited CPTs → NetworkSpec (per-CPT κ) → pgmpy engine + dashboard│
├──────────────────────────────────────────────────────────────────┤
│ Layer 4 — UI (Streamlit)                          ── v1 (0–4) ──   │
│  Cooke workflow · CPT review · Sources · HITL · calibration       │
├──────────────────────────────────────────────────────────────────┤
│ Layer 3 — Multi-model panels & defensibility hardening            │
│  decorrelation · contamination probes · calibration reports · red-team│
├──────────────────────────────────────────────────────────────────┤
│ Layer 2 — Cooke elicitation + AI experts   (it runs)              │
│  Cooke state machine · Expert interface · LLMExpert · seed scoring │
├──────────────────────────────────────────────────────────────────┤
│ Layer 1 — Core engine                                             │
│  Cooke aggregation · sensitivity · calibration→κ mapping          │
├──────────────────────────────────────────────────────────────────┤
│ Layer 0 — Data model and storage substrate                        │
│  Multi-deployment isolation · schema · auth · audit-log extension │
└──────────────────────────────────────────────────────────────────┘
```

### Data model

Each deployment runs an isolated stack. Within a single deployment:

| Table | Purpose |
| --- | --- |
| `users`, `roles`, `permissions` | RBAC primitives |
| `networks` | This deployment's DAGs |
| `cpts` | CPT current values, indexed by network + node |
| `cpt_versions` | Historical CPT values with full audit trail |
| `cpt_provenance` | Per-CPT-version metadata: protocol, elicitor(s), date, references, **κ value and level, calibration score, model set, correlation note, contamination-probe summary** |
| `experts` | Registered experts (human or AI), with calibration history. An AI expert's identity is the tuple `(base_model, role, config)` — see Layer 2/3 — stored as columns here |
| `expert_calibration` | Per-expert (i.e. per `(base_model, role, config)`) performance on seed questions |
| `provider_credentials` | LLM provider credentials for this deployment: source (`deployment_key` / `oauth` / `byok`), provider, owning user (for BYOK), envelope-encrypted secret reference (never the plaintext key), rotation/revocation state |
| `seed_sets` | Per-deployment calibration question sets, with resolution dates and source provenance |
| `contamination_checks` | Per-AI-expert, per-seed probe results (source-attribution, perturbation, in-corpus split) |
| `elicitation_sessions` | Protocol runs with all inputs and aggregated outputs |
| `elicitation_session_events` | State-machine events for resumable workflows |
| `outcomes` | Realised intermediate-node states for Tier 2/3 calibration |
| `calibration_runs` | Scheduled evaluation results |
| `articles`, `translations`, `analyst_actions`, `sources` | Shared with Plan 2 |

No `tenant_id` anywhere. Storage: SQLite for development, Postgres for production. ORM: SQLAlchemy / SQLModel. Migrations: Alembic.

## Section A — Layered plan

### Layer 0 — Data model and storage substrate

**Status.** ⬜
**Resolves.** Diagnosis 4 (versioning), 5 (provenance), 9 (reuse across deployments).

**Scope.** Define and ship the elicitation schema as an extension of Plan 2's audit-log schema. Per-deployment isolated database. Auth scaffolding (per-deployment user table, hooks for SSO). Per-deployment YAML configuration (network choice, active topology, in-scope CPTs, branding). **LLM provider credentials** — a `CredentialResolver` abstraction over three sources (deployment key, OAuth-brokered, and policy-gated bring-your-own-key) feeding a single `ProviderCredential` interface the agents consume; secrets are envelope-encrypted at rest (KMS / secret manager, never plaintext in the DB), deployment-scoped, never logged, and redacted from all provenance and audit trails (the *model identity* is recorded; the *key* never is).

**Deliverables.**
- `src/elicitation/db/schema.py` — SQLAlchemy / SQLModel definitions (all tables above, incl. `provider_credentials`).
- `src/elicitation/db/migrations/` — Alembic migrations.
- `src/elicitation/auth/` — per-deployment user/role tables and login plumbing.
- `src/elicitation/config/` — per-deployment YAML configuration loader, including the BYOK provider allowlist (per-deployment policy; may be empty to disable BYOK for high-sensitivity tiers).
- `src/elicitation/credentials/` — `CredentialResolver`, `ProviderCredential`, the secret-store adapter (envelope encryption), and rotation/revocation plumbing.
- `docs/deployment.md` — how to stand up a new deployment.

**Validation.**
- Schema migrates cleanly on a fresh SQLite database and a fresh Postgres database.
- Two parallel deployments operate with no cross-database visibility; credentials in one are never resolvable from the other.
- A BYOK key is never present in any log, audit row, or provenance record (redaction test).
- BYOK is refused for a provider not on the deployment's allowlist.
- Migration round-trip (apply then roll back) leaves the database in its pre-migration state.

### Layer 1 — Core engine: Cooke aggregation, sensitivity, calibration→κ

**Status.** ⬜
**Resolves.** Diagnosis 6 (calibrated κ), 7 (sensitivity prioritisation). Provides the mathematical primitives the protocol and the inference integration consume. Methodology: [§4–§6](elicitation_methodology_and_defensibility.md).

**Scope.** Pure-function library, no database or UI dependencies. Three subsystems:

1. **Aggregation primitives.** Linear and logarithmic (geometric) pooling, and the performance-weighted Cooke pool built on the linear pool — each a pure function over expert distributions. (Linear/log are retained as primitives because `cooke_pool` is built on the weighted linear pool and the future IDEA/SHELF protocols reuse them.)
2. **Sensitivity analysis.** Morris screening (cheap, qualitative) and Sobol indices (quantitative) wrapping SALib, plus a variance decomposition of the propagated posterior that ranks each CPT's contribution to output-interval width. This is the **prioritiser** that directs review and κ-tightening effort (Layer 6), not a gate on whether to elicit.
3. **Calibration→κ mapping.** Functions that turn measured calibration and panel disagreement into a per-CPT Dirichlet concentration `κ`:
   - `kappa_from_panel_spread(expert_vectors, correlation)` — method-of-moments estimate of `κ` from the dispersion of the panel's point vectors, discounted by measured inter-agent correlation (effective sample size).
   - `kappa_from_seed_coverage(seed_predictions, truths)` — fits `κ` so the panel's predictive intervals attain correct empirical coverage on the seed set.
   - `snap_to_level(kappa, levels)` — rounds a continuous `κ` to the nearest ordinal level in `{tight, normal, uncertain}` (per-deployment fitted values); `cap_level(level, calibration_score)` — caps the level an expert may contribute by its measured calibration.

**Deliverables.**
- `src/elicitation/engine/aggregation.py` — `linear_pool`, `logarithmic_pool`, `cooke_pool`.
- `src/elicitation/engine/sensitivity.py` — `morris_screening`, `sobol_indices`, `posterior_variance_decomposition`.
- `src/elicitation/engine/kappa.py` — the calibration→κ functions and the `{tight, normal, uncertain}` ladder.
- `tests/elicitation/test_engine.py` — unit coverage against published reference outputs.

**Validation.**
- Linear and logarithmic pooling agree on degenerate inputs (single expert) and match hand-computed two-expert cases.
- `cooke_pool` reproduces a published classical-model worked example.
- Morris/Sobol outputs agree with SALib's reference outputs on a small test BN.
- `kappa_from_seed_coverage` recovers a known `κ` on synthetic data sampled from a fixed Dirichlet.
- `kappa_from_panel_spread` widens (lowers `κ`) monotonically as injected inter-agent correlation rises.

### Layer 2 — Cooke elicitation + AI experts

**Status.** ⬜
**Resolves.** Diagnosis 8 (formal protocol), 2 (multi-expert aggregation), finding M6 (root priors elicited, not asserted). The minimal runnable elicitation pipeline. Methodology: [§3, §8](elicitation_methodology_and_defensibility.md).

**Scope.** The Cooke workflow as a resumable state machine, plus the AI-expert capability that drives it in the AI-only operating mode. Because the panel is AI-only, the single-agent `LLMExpert` is **part of this layer** — without it there is no expert input at all.

The protocol exposes:
- `required_experts() -> tuple[int, int]` — min/max expert count (Cooke: 4–12).
- `workflow() -> WorkflowSpec` — the ordered sequence of steps (seed elicitation → target elicitation, scoring deferred to aggregation).
- `aggregate(expert_inputs) -> CPTColumn` — seed-scored, performance-weighted linear pool; experts below the calibration cutoff are zeroed.
- `provenance_record() -> ProvenanceMetadata` — what is written to `cpt_provenance` at conclusion.

**Cooke runs once per panel, reused across all CPTs.** Seed scoring — the expensive step — is performed **once per panel per deployment**; the resulting weights are reused across every CPT column that panel elicits. There is no per-node "is this worth Cooke?" gate: the default is to elicit **every** in-scope CPT. Sensitivity analysis (Layer 1/6) prioritises review and κ-tightening effort, not whether elicitation happens.

**Targets.** Each `workflow()` is parameterised by an `ElicitationTarget`. The single in-scope shape is a CPT column — `CPTColumnTarget`, eliciting one $P(Y \mid \text{Pa}(Y) = u)$, a categorical distribution on $|Y|$ states. The base class keeps the protocol target-agnostic so additional shapes can be added later.

**AI expert (single agent, this layer).** `LLMExpert` implements the *same* per-step `Expert` interface a human uses: answer a seed quantile question, answer a target distribution, etc. It obtains its provider access from the Layer 0 `CredentialResolver` (`ProviderCredential`) and is provider-agnostic. Seed scoring (Layer 1) applies identically to it. The *defensibility hardening* of the AI panel — multi-model diversity, decorrelation, contamination probes — is Layer 3; this layer establishes that a scored agent can drive Cooke end-to-end.

**Expert identity — `(base_model, role, config)`.** An AI expert is not just a model. Its identity is the tuple `(base_model, role, config)`, and **calibration is measured per tuple**, because a role-conditioned estimate is a different estimator: an agent seed-scored "neutral" but answering targets "in role" carries no valid calibration. Two rules follow. (i) Seed scoring and target elicitation must run in the **same** configuration, so a role used to set scored estimates is seed-scored in that role. (ii) Roles whose purpose is to surface considerations rather than to set the scored probability (e.g. a brainstorm/divergence pass) must not feed the scored estimate directly. Base-model diversity — not role/persona variety — is what counts for decorrelation (Layer 3, methodology §8.2).

**Deliverables.**
- `src/elicitation/protocols/base.py` — `ElicitationProtocol`, `Expert` (human/AI), `WorkflowSpec`, `ProvenanceMetadata`, `ElicitationTarget`.
- `src/elicitation/protocols/targets.py` — `CPTColumnTarget` with validation and aggregation hooks.
- `src/elicitation/protocols/cooke.py` — `CookeProtocol`, seed-scoring (statistical accuracy × information), performance-weighted aggregation with cutoff zeroing.
- `src/elicitation/agents/llm_expert.py` — single-agent `LLMExpert` on the `Expert` interface, identity `(base_model, role, config)`, consuming `ProviderCredential`.
- `tests/elicitation/test_cooke.py` — end-to-end Cooke run with synthetic and `LLMExpert` inputs; seed scoring down-weights a poor expert and zeroes one below the cutoff; provenance is written.

**Validation.**
- Cooke runs end-to-end on a test network with an `LLMExpert` panel, producing a CPT with attached provenance.
- The state machine persists across simulated restarts (kill mid-workflow, resume from the database).
- Aggregation matches published classical-model references for canonical small examples.
- An agent that fails the seeds is down-weighted; one below the cutoff is zeroed.

### Layer 3 — Multi-model panels & defensibility hardening

**Status.** ⬜
**Resolves.** Extends diagnosis 1, 2, 3, 10 to make the AI-only panel **defensible**, not merely runnable. Methodology: [§8.2–§8.7](elicitation_methodology_and_defensibility.md).

**Scope.** Everything that turns a single scored agent (Layer 2) into a defensible multi-model panel. This is *not* optional polish: single-model naive pooling is the central AI-elicitation failure mode (correlated error, contamination), so this layer ships immediately after the core.

- **Multi-model orchestration.** Panels composed of genuinely different base models (not personas/temperatures of one). IDEA-style round-1-before-exposure discipline is preserved even for Cooke discussion-free runs.
- **Roles and characters (additive, not a substitute for diversity).** Each base model may be assigned a role/persona — most importantly a red-team agent prompted to refute, plus optional perspective roles (base-rate thinker, escalation-pessimist, etc.) to widen consideration coverage. Roles compose with base-model diversity; they **do not** count toward independence (five personas on one model is still one correlated source — methodology §8.2). Each `(base_model, role, config)` is a distinct scored expert (Layer 2), so roles never bypass the calibration gate, and a role used to set scored estimates is seed-scored in that role.
- **Decorrelation.** Estimate inter-agent correlation **across base models**; shrink the effective sample size and widen κ accordingly (consumes Layer 1's `kappa_from_panel_spread`). Roles/personas are excluded from the independence count.
- **Contamination probes.** Source-attribution, perturbation/canary, in-corpus-vs-post-cutoff split scoring, and anomalously-low cross-model variance as a leakage alarm. Results written to `contamination_checks`. Seed calibration is treated as a **filter that removes poor agents, never a certificate of trust**; primary calibration is prospective (Layer 6 Tier 2/3), retrodictive post-cutoff seeds are a flagged bootstrap.
- **Calibration reports.** Per-agent seed scores + panel-level report (model set, correlation, contamination summary), written to `cpt_provenance`.
- **Judge independence.** Where an agent judges another agent, the judge must be a different base model; no self-grading.

**Deliverables.**
- `src/elicitation/agents/panel.py` — multi-model panel orchestration; Cooke runner over diverse LLM experts; role/persona assignment (red-team + optional perspective roles), with each `(base_model, role, config)` registered as its own scored expert.
- `src/elicitation/agents/decorrelation.py` — correlation estimation and effective-weight/κ adjustment.
- `src/elicitation/agents/contamination.py` — the probes; writes `contamination_checks`.
- `src/elicitation/agents/calibration_report.py` — per-agent + panel report, written to `cpt_provenance`.
- `tests/elicitation/test_panel.py` — same-model correlation is detected and lowers effective weight; a contaminated seed is caught by perturbation; leave-one-seed-out cross-validation does not degrade the aggregate versus equal weight (the Clemen test, applied to AI experts).

**Validation.**
- Same-model agents show high measured correlation; cross-model agents lower — effective weight and κ adjust accordingly.
- Two roles on the **same** base model are *not* credited as independent: their combined effective weight stays ~1, not 2.
- A role-conditioned expert seed-scored "neutral" is rejected from contributing scored estimates in-role (configuration-mismatch guard).
- A planted contaminated seed is flagged by at least one probe.
- A fully-automated panel's CPT lands within a stated tolerance of a reference CPT on a back-test set, with calibration reported.
- Every AI-sourced CPT is flagged in `cpt_provenance` with calibration score, κ level, model set, correlation note, and contamination summary — a hard defensibility requirement.

### Layer 4 — UI (Streamlit)

**Status.** ⬜
**Resolves.** Makes Layers 0–3 usable by non-engineer analysts. Completes the **v1 milestone**.

**Scope.** Streamlit pages exposing the Cooke workflow and review surfaces:
- **New CPT elicitation.** Pick a node, run the Cooke wizard step-by-step. State persists between page loads.
- **CPT review and override.** Inspect a current CPT, compare to historical versions, override a cell (recorded as a manual edit in `cpt_provenance`).
- **Sources tab.** Shared with the translator (Plan 2 B1b).
- **HITL triage.** Shared with the translator (Plan 2 E1).
- **Calibration dashboard.** Per-expert weight history; per-CPT κ level (`tight`/`normal`/`uncertain`); panel model-set, role assignments, and correlation/contamination summaries.
- **CPT version history viewer.** Time-machine view of any CPT across the sessions that produced it.
- **Provider settings.** A per-user surface to add a BYOK key (provider from the deployment allowlist only) and a per-deployment view of configured credential sources. Keys are write-only from the UI (never displayed back).

**Credential source is transparent to the UI.** The dashboard resolves LLM access through the Layer 0 `CredentialResolver` and is agnostic to whether the credential is a deployment key, OAuth, or BYOK — so *whatever is passed, the dashboard works*. The existing pgmpy inference and credible-interval views have **no LLM dependency** and must remain fully functional when no LLM credential is present; only the elicitation/proposal surfaces degrade gracefully (clear "add a provider credential" prompt) in that case.

**Deliverables.**
- `app/elicitation/` — Streamlit pages, including the provider-settings surface.
- `app/elicitation/components/` — distribution editors, quantile/roulette pickers, calibration plots, version diff views, κ-level badges.
- `app/elicitation/styles.css`.

**Validation.**
- The Cooke workflow runs end-to-end in the UI under each credential source (deployment key, OAuth, BYOK).
- With **no** LLM credential configured, the pgmpy dashboard and credible-interval views still render; elicitation surfaces show a graceful prompt rather than erroring.
- A BYOK key entered in the UI is write-only (never rendered back) and refused for a non-allowlisted provider.
- Auth and per-deployment isolation work (login, session management, no cross-deployment leakage).
- Walkthrough documentation exists (`docs/elicitation_walkthroughs.md`).

### Layer 5 — Inference integration

**Status.** ⬜
**Resolves.** Finding M3 (per-CPT κ carried from provenance into the engine). Couples elicited CPTs to inference and the live dashboard.

**Scope.** Elicited CPTs export to a declarative `NetworkSpec`; round-trip elicit → save → load → run inference. Ships the **discrete** subset of `NetworkSpec` (`DiscreteNode`, `NetworkSpec`) in `src/network_spec.py`, authored so the later PyMC extension extends the same file.

**Per-CPT κ — the concrete replacement of the global constant.** Today [`src/sensitivity.py`](../src/sensitivity.py) resamples every CPT with one scalar `concentration` (`_resample_cpd`). This layer changes `_resample_cpd` to accept a **per-CPT κ** read from `cpt_provenance` (via `DiscreteNode.kappa`), so the credible intervals the dashboard already renders become calibration-grounded rather than a uniform guess. Back-compatible: callers passing a single κ keep the old behaviour.

```python
# src/elicitation/export/network_spec.py
def cpts_to_network_spec(network_id: int, snapshot_at: datetime | None = None) -> NetworkSpec:
    """Build a NetworkSpec from this deployment's CPTs.

    If snapshot_at is given, uses the CPT versions in force at that time;
    otherwise current versions. Each DiscreteNode carries its per-CPT kappa
    from cpt_provenance.
    """
```

The inverse (`network_spec_to_cpts`) is supported, so an existing `NetworkSpec` (e.g. the bootstrap Hormuz network in `src/network.py`) can be imported as a refinement starting point.

**Deliverables.**
- `src/network_spec.py` — discrete `NetworkSpec` (`DiscreteNode` with `kappa`, `NetworkSpec`, validation).
- `src/elicitation/export/network_spec.py` — `cpts_to_network_spec`, `network_spec_to_cpts`.
- Patch to `src/sensitivity.py` — `_resample_cpd` accepts per-CPT κ; back-compatible scalar default.
- `src/elicitation/export/cli.py` — command-line export/import.
- Integration tests for the round trip.

**Validation.**
- Round-trip: load Hormuz from `src/network.py` → store → export back → identical `NetworkSpec`.
- Per-CPT κ flows into `DiscreteNode.kappa` and the resampling path; a CPT marked `uncertain` produces a visibly wider interval than one marked `tight`.
- Snapshot-at-time queries return the historically-correct CPT version.
- The pgmpy engine produces posteriors matching the current baseline when κ is uniform.

### Layer 6 — Calibration tracking & confidence over time

**Status.** ⬜
**Resolves.** Diagnosis 1 (hardcoded → elicited and *validated*), 3 (calibration tracking), 10 (translator coupling). This is the old "advanced features" layer, narrowed to one coherent idea: **how the calibrated numbers improve and get reported honestly over time.** Methodology: [§6–§7](elicitation_methodology_and_defensibility.md).

**Scope.** Five subsystems, in priority order:

1. **Confidence reporting (the payoff of the κ work).** Assembles the defensible final-outcome confidence statement ([§7](elicitation_methodology_and_defensibility.md)): point posterior, propagated credible interval, the variance decomposition, the empirical calibration track record (where it exists), and an IPCC-style confidence rating with an explicit structural-uncertainty caveat. A reporting layer over the existing propagation, not a new inference path. The *prospective* components are available as soon as Layer 5 lands; the empirical track record accrues with subsystems 3–4.
2. **Sensitivity-driven prioritisation workflow.** Wraps Layer 1's variance decomposition in an analyst-facing view: which CPT entries dominate the posterior under current evidence, and where tightening κ would most reduce the output interval. Directs re-elicitation effort.
3. **Calibration Tier 2 — intermediate-node tracking.** For nodes whose outcomes can be observed, record the realised outcome in `outcomes`, compute Brier scores and reliability diagrams over time, and refine per-CPT κ via empirical coverage updating.
4. **Calibration Tier 3 — Bayes factor / regime trajectory.** For the latent-regime model, record log-Bayes-factor predictions for each evidence increment against expert-judged "true" regime trajectories on historical analogs. Reveals systematically miscalibrated CPT regions.
5. **(Optional) LLM-proposed initial CPT values.** Couples to Plan 2's RAG memory (E2): the LLM retrieves relevant analog events and proposes initial values with span-grounded citations for an expert to review, edit, or reject. The proposal commits nothing.

Subsystems 3–4 feed back into Cooke (Layer 2): experts accrue calibration scores as their contributions' outcomes are observed, and their weights — and the contamination-aware κ caps — update accordingly.

**Deliverables.**
- `src/elicitation/reporting/confidence.py` — the assembled confidence statement.
- `src/elicitation/sensitivity/workflow.py` — analyst-facing prioritisation.
- `src/elicitation/calibration/tier2.py`, `tier3.py` — outcome tracking, Brier/reliability, Bayes-factor trajectories.
- `src/elicitation/calibration/expert_weights.py` — weight and κ-cap updates from accrued calibration.
- `src/elicitation/proposals/llm.py` — RAG-augmented proposal generator (consumes Plan 2's E2 index).

**Validation.**
- The confidence report renders all components and never collapses them to a single scalar.
- Prioritisation ranks CPT entries in agreement with Sobol indices on a test network.
- Tier 2 data accumulates over a simulated history; Brier scores and reliability diagrams render correctly.
- Tier 3 replays produce Bayes-factor trajectories matching expert-judged trajectories within calibrated noise.
- Cooke weights and κ caps update sensibly as accrued predictions are scored against outcomes.

## Section B — Design decisions

All decisions below are resolved. Arguments and references: [methodology doc](elicitation_methodology_and_defensibility.md).

1. **Deployment shape — multi-deployment, single-tenant.** Own isolated stack per deployment; no `tenant_id`.
2. **UI — Streamlit for v1**, upgrade later when scaling justifies it.
3. **Storage — SQLite (dev), Postgres (prod)**; SQLAlchemy/SQLModel; Alembic.
4. **Audit-log substrate — extend Plan 2's schema**, not duplicate.
5. **`NetworkSpec` — this plan ships the discrete subset**; integrates with the existing pgmpy engine; the PyMC extension (Plan 3) extends the same file.
6. **Protocol — Cooke only is built.** Cooke is the sole implementation target. IDEA and SHELF are documented (methodology §3) and retained behind the same `Protocol`/`Expert` interface for the future, but out of scope. No per-node elicitation gate: Cooke is scored once per panel and applied to all in-scope CPTs.
7. **AI experts are on the critical path, hardened immediately after.** Because panels are AI-only, the single-agent `LLMExpert` is part of the elicitation core (Layer 2); multi-model diversity, decorrelation, and contamination probes follow directly (Layer 3) as a defensibility requirement, not a late add-on.
8. **Per-CPT κ from calibration, reported on a three-level ordinal ladder** (`tight`/`normal`/`uncertain`). κ is estimated from panel disagreement (correlation-discounted) and/or seed coverage, then snapped to a level; an expert's measured calibration caps the level it may contribute. Replaces the current global `κ = 20`.
9. **Confidence is reported as a vector, never a scalar** — point posterior + propagated credible interval + variance decomposition + empirical calibration track record + IPCC-style confidence rating + structural caveat.
10. **Calibration tiers — three, phased.** Tier 1 (translator-level) via Plan 2 D2. Tier 2 (intermediate-node) from Layer 6; signal accumulates over months. Tier 3 (Bayes factor / regime trajectory) once the latent regime is in production.
11. **Translator coupling — deep.** Audit log → analog events for anchored elicitation; RAG memory → optional LLM proposals; HITL queue → elicitation proposals.
12. **Seed set — per-deployment, relevance-constrained.** Each deployment builds its own domain-relevant seed set, reused across all Cooke runs. Seeds must probe the same judgment as the targets.
13. **AI-only panels (v1), human-capable interface.** Panel members are multi-model AI agents; human experts remain first-class via the identical `Expert` interface and are enabled per deployment. Human sign-off for high-stakes is mandatory.
14. **AI-expert calibration — prospective-primary, retrodictive-bootstrap, probe-gated.** Seed scoring filters out bad agents; it does not license trust. Primary calibration is prospective (Tier 2/3); retrodictive post-cutoff seeds are a flagged bootstrap with active contamination probes. Panels must be multi-model; every AI-sourced CPT is attributed in `cpt_provenance`.
15. **LLM provider credentials — three sources behind one interface.** A `CredentialResolver` (Layer 0) serves `ProviderCredential` from a deployment key, OAuth, or **policy-gated bring-your-own-key**. Secrets are envelope-encrypted, deployment-scoped, never logged, and redacted from all provenance (the model identity is recorded; the key never is). BYOK is restricted to a per-deployment provider allowlist and may be disabled entirely for high-sensitivity deployments — because a BYOK key routes the source material to that provider under the user's terms, which is a data-residency decision, not just a security one. The dashboard is agnostic to the credential source and its non-LLM views work with no credential present.
16. **AI experts are `(base_model, role, config)` tuples; roles are additive.** Each tuple is a distinct scored expert with its own calibration record. Roles/personas (red-team, perspective roles) widen consideration coverage and compose with base-model diversity, but **do not** count toward independence in decorrelation, and never bypass the calibration gate — a role used to set scored estimates is seed-scored in that role.

## Section C — Open questions

These do not block Layer 0 but should be resolved before the corresponding layer begins.

| Question | Block | Notes |
| --- | --- | --- |
| Per-deployment Postgres setup automation | Layer 0 | Docker Compose for the MVP; Terraform/Helm for production. |
| Auth provider for SSO | Layer 0 | Self-hosted (Keycloak, Authentik) vs hosted (Auth0, Clerk). Recommend hosted for v1. |
| Open-source license | Layer 0 | If released: recommend Apache 2.0 for its patent grant. |
| Secret-store backend & BYOK allowlist | Layer 0 | Which KMS / secret manager for envelope encryption (cloud-managed vs self-hosted, e.g. Vault); and the per-deployment provider allowlist policy, including the high-sensitivity "BYOK disabled" tier. |
| Role/persona catalogue | Layer 3 | Which roles beyond red-team add real consideration coverage, and whether any role-conditioned estimates feed scored CPTs (vs roles confined to a divergence pass). Validate that roles improve panel calibration rather than just adding correlated noise. |
| Three-level κ ladder — fitted values | Layer 1 | Fit `tight`/`normal`/`uncertain` κ per deployment on the seed set. Validate the snap-and-cap logic against held-out coverage. |
| Categorical coverage-fit for κ | Layer 1 | Natural for continuous seeds; the categorical-CPT analog (proper-scoring-based) needs specifying. See methodology §6. |
| Cooke seed-set design | Layer 2 | Hormuz-specific seeds need authoring with domain input; relevance-constrained. Defer until the first Cooke deployment is concrete. |
| LLM provider & model diversity | Layer 2 / 3 | Coupled to Plan 2's translator provider choice; panels require ≥2 distinct base models. How many, and how to estimate/adjust correlation. |
| Leakage-free prospective seed pipeline | Layer 3 / 6 | Rolling, post-cutoff seeds plus the prospective-scoring loop; defines how AI-expert calibration stays un-gameable across model upgrades. |
| UI upgrade trigger | Layer 4 | Define the criterion (concurrent users per deployment, custom components needed). |

## Section D — Execution order

| Order | Layer | Resolves | Rationale |
| --- | --- | --- | --- |
| 1 | 0 — Data model and storage | Diagnosis 4, 5, 9 | Per-deployment isolation, versioning, provenance, auth. Unblocks everything. |
| 2 | 1 — Core engine | Diagnosis 6, 7 | Cooke aggregation, sensitivity, calibration→κ. Pure-function; fully unit-testable. |
| 3 | 2 — Cooke elicitation + AI experts | Diagnosis 8, 2, finding M6 | Cooke state machine + single-agent `LLMExpert`. The minimal runnable pipeline. |
| 4 | 3 — Multi-model panels & hardening | Diagnosis 1, 2, 3, 10 (AI) | Decorrelation, contamination probes, calibration reports. Makes the AI panel defensible. |
| 5 | 4 — UI (Streamlit) | Usability | **v1 milestone**: usable, defensible, AI-only Cooke elicitation. |
| 6 | 5 — Inference integration | Finding M3 | Elicited CPTs → `NetworkSpec` (per-CPT κ) → pgmpy + dashboard. Replaces the global κ. |
| 7 | 6 — Calibration & confidence over time | Diagnosis 1, 3, 10 | Confidence report, prioritisation, calibration Tiers 2/3, weight/κ updates, optional LLM drafts. |

## Future directions

- **Additional protocols (documented, not built).** IDEA and SHELF are specified in the [methodology companion](elicitation_methodology_and_defensibility.md) §3 and slot in behind the `Protocol`/`Expert` interface when a mid- or moderate-stakes deployment needs them. No restructuring required.
- **Ranked-node CPT compression (out of scope).** The Fenton & Neil ranked-node method collapses a high-fan-in *ordinal* CPT to a few per-parent weights. Deliberately left out: the current Hormuz topology has no CPT large enough to need it (emission CPTs are single-parent; the latent regime is categorical, so ineligible — ranked nodes require an ordinal child). If a future engagement involves high-fan-in ordinal nodes, add it as an optional Layer 1 engine primitive first, validated against the Fenton & Neil worked examples. Applicability is gated on monotonic parent→child relationships.
- **Additional `ElicitationTarget` shapes.** The base class exists so shapes beyond `CPTColumnTarget` can be added without restructuring — most notably a temporal `TransitionMatrixTarget` if a BN↔HMM extension is ever built ([`docs/bn_hmm_integration.md`](bn_hmm_integration.md)).

---

**End of plan.** Methodology, design rationale, defensibility argument, and references: [`docs/elicitation_methodology_and_defensibility.md`](elicitation_methodology_and_defensibility.md). Companion plans: [`docs/01_latent_regime_plan.md`](01_latent_regime_plan.md) (topology), [`docs/02_translator_robustification.md`](02_translator_robustification.md) (substrate), [`docs/03_pymc_integration_plan.md`](03_pymc_integration_plan.md) (PyMC inference extension).
