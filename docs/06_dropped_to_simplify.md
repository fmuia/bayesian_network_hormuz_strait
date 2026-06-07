# Dropped to Simplify the First Implementation

> **Purpose.** A single, canonical parking lot for capabilities deliberately **taken out of the first implementation to streamline it**. Nothing here is rejected on the merits — each item has a preserved design, a "what we do instead for now" substitute, and an explicit **re-introduction trigger**. When a trigger fires, lift the item out of this doc and back into its home plan.
>
> **Why this exists.** Plans 1–5 describe the full platform. Building all of it at once is not the goal of the first pass. This file is where the orchestrator records *what was consciously deferred and why*, so a deferred feature never silently becomes an assumed-present one (and so the plans that reference it don't drift).
>
> **Status legend.** 🅿️ parked · ♻️ re-introduced (with date). Every item below is 🅿️.

## How to use this file

- Each entry names **what** was dropped, **where it lived** (plan + commit/section anchors), **why** it was dropped, the **substitute** in force now, the **re-introduction trigger**, and the **findings affected** (if any).
- The home-plan docs link *here* at the point of removal rather than describing the deferral inline — this file is the source of truth for "is X in or out right now."
- Re-introducing an item is a deliberate act: update the trigger to ♻️ with a date, restore the home-plan text, and open the work.

---

## Summary

| # | Dropped item | Home plan | Substitute now | Re-introduction trigger |
|---|--------------|-----------|----------------|-------------------------|
| **1** | Embedding-based translator features (relevance pre-filter, paraphrase dedup, retrieval index) | Plan 2 (B3/T05, B2/T06a, E2/T13) | LLM-only equivalents / deferred reach item | Throughput or calibration data shows the LLM-only path is insufficient |
| **2** | PyMC / continuous-variable inference backend (dual backend, hierarchical priors, continuous Oil_Price, per-CPT κ) | Plan 3 (all phases) | Stay on the existing pgmpy discrete path | A continuous variable or hierarchical-prior calibration becomes necessary |
| **3** | Plan 2 institutional layer (ensemble, prompt versioning, sqlite provenance, source-credibility history, multi-model cross-check, RAG, riders) | Plan 2 (T07, T08, T09, T10, T11, T13, R-judge/R-cal/R-pair) | POC ships T00–T06 + a slim in-session HITL (T12); in-session audit + saved sessions cover provenance | Moving from POC to a paying/production engagement |

---

## 1. Embedding-based translator features  🅿️

**Where it lived.** Plan 2 — [`docs/02_translator_robustification.md`](02_translator_robustification.md) §B2, §B3, §E2, design decision 9; commit plan [`docs/02_translator_robustification_commit_plan.md`](02_translator_robustification_commit_plan.md) T05, T06a, T13 (§6 D1).

**What was dropped.** Three sub-features that all assumed a sentence/document **embedding provider**:

1. **B3 / T05 — embedding relevance pre-filter.** A topic-anchor embedding-distance check that rejects off-topic articles *before* the expensive LLM call.
2. **B2 / T06a — embedding-cosine paraphrase dedup.** Merging claims whose `verbatim_span` embeddings have cosine ≥ 0.9 (design decision 9), to prevent paraphrase double-counting in the multiplicative aggregation.
3. **E2 / T13 — retrieval embedding index.** Indexing the audit log by article embedding for top-K semantic retrieval of analyst-approved precedents.

**Why dropped.** Claude exposes no embeddings API, so an embedding provider would be a *new* external dependency (OpenAI embeddings) or a heavyweight local one (`sentence-transformers`/torch) — neither justified for the first pass. On re-examination the early features did not actually *need* embeddings:

**Substitute in force now (LLM-only):**

- **Relevance (T05):** `relevance ∈ {yes, partial, no}` is produced directly as an LLM field. The embedding pre-filter was only a *cost optimization* (skip the LLM on obvious junk), which does not pay for itself at demo cadence (≤10 articles/day).
- **Dedup (T06a):** done by **prompt discipline** — the claim-extraction step is instructed to emit atomic, mutually-distinct claims (the same fact never listed twice, even if rephrased). The `verbatim_span` substring check still rejects ungrounded claims, and C1 disagreement + HITL catch residual paraphrase-dups. *This is the LLM-only form of design decision 9.*
- **Retrieval (T13):** deferred entirely — T13 is the last, most-deferred reach item. When undertaken, small-corpus **lexical (BM25) or LLM-mediated** retrieval is the fallback before any embedding index.

**Re-introduction trigger.**
- T05 pre-filter: daily throughput rises past demo cadence such that skipping LLM calls on off-topic input is worth a dependency.
- T06a embedding backstop: D2 calibration shows paraphrase double-counting is *materially* degrading accuracy despite prompt-discipline dedup.
- T13 embedding index: T13 is picked up *and* the corpus has grown past the size where lexical/LLM retrieval is adequate. At that point, resolve the embedding-provider choice (OpenAI `text-embedding-3-*` vs local `sentence-transformers`).

**Findings affected.** None — these were enhancements, not M/C/V findings. No coverage is lost from the master-plan matrix.

---

## 2. PyMC / continuous-variable inference backend  🅿️

**Where it lived.** Plan 3 — [`docs/03_pymc_integration_plan.md`](03_pymc_integration_plan.md) (all phases 0–4); referenced across [`docs/master_plan.md`](master_plan.md) §2 (architecture), §3 (plan sequence), §4 (findings matrix), and the decision log (architecture decisions 1–2).

**What was dropped (executive decision — PyMC not on the current roadmap).**

- **Dual-backend architecture** (`PgmpyBackend` / `PymcBackend`) and the declarative `NetworkSpec` that dispatches between them.
- **Hierarchical priors over CPTs** (PyMC-native).
- **Continuous variables**, including the planned `Oil_Price` migration from the discrete `Oil_Price_Regime` to a continuous node.
- **Per-CPT κ** as an operative parameter (it is consumed by `PymcBackend`'s hierarchical priors).

**Why dropped.** The first implementation does not need continuous variables or hierarchical priors; the existing pgmpy discrete path is sufficient for Plans 1–2 and the dashboard. Adding PyMC is a large engine investment whose payoff (continuous Oil_Price, hierarchical CPT uncertainty) is not yet required.

**Substitute in force now.** Everything stays on the **existing pgmpy discrete code path** (`src/network.py`, `src/inference.py`). Plan 1's latent-regime topology already ships there. `Oil_Price_Regime` stays discrete. Plan 1's uniform κ = 10 remains a *provisional, inert* value — pgmpy variable elimination never consults κ, so the discrete path is insensitive to it.

**Re-introduction trigger.** A stakeholder need forces a **continuous** variable (e.g., a literal oil-price level rather than a 3-band regime), **or** calibration work requires **hierarchical priors** over CPTs that the bolt-on Dirichlet resampling cannot express. At that point Plan 3 is lifted back out of this doc; its full design is preserved verbatim in `03_pymc_integration_plan.md`.

**Findings affected (NOT closed while parked).** These master-plan §4 findings depended wholly or partly on Plan 3 and are therefore **not closed** in the first implementation:

- **M2 (soft evidence semantics) — continuous-node facet.** The *translator-interface* facet of M2 is still closed by Plan 2 A1 (discrete); only the continuous-node facet (Plan 3 Phase 3) is parked.
- **M3 (uniform κ).** The per-CPT κ pathway (Plan 3 Phase 3 + Plan 4 Layer 4) is parked; κ stays uniform and inert.
- **M4 (independent CPT column resampling).** The hierarchical-priors fix (Plan 3 Phase 3) is parked; the bolt-on Dirichlet resampling remains.

The master-plan matrix rows for M2/M3/M4 are annotated to point here.

---

## 3. Plan 2 institutional layer (post-POC)  🅿️

**Where it lived.** Plan 2 commit plan [`docs/02_translator_robustification_commit_plan.md`](02_translator_robustification_commit_plan.md): **T07** (C1 self-consistency ensemble), **T08** (D1 prompt versioning + pre-commit gate), **T09** (D3 sqlite provenance / reproducibility / body retention), **T10** (B1b per-source credibility editing + history), **T11** (C2 multi-model cross-check), **T13** (E2 retrieval-augmented translation), and the riders **R-judge / R-cal / R-pair**.

**What was dropped (skeptical-gate decision, 2026-06-08).** After T06 the translator already makes a compelling stakeholder demo (likelihood-ratio semantics, hardened schema, article + source credibility, relevance/abstention, optional span-grounded + injection-resistant structured reasoning, an eval badge, an in-session audit trail). The remaining items are the plan's own self-described "institutional layer that distinguishes a tool from a script" — productization, not demo. The POC therefore ships **T00–T06 + a slim, in-session HITL (T12)** and parks the rest here.

**Why dropped for the POC.**
- **T07 ensemble** — 5–10× LLM cost; the dashboard's CPT-resampling credible intervals already carry an uncertainty story. (Cheap substitute if ever wanted: a verbalised-confidence field in the existing call.)
- **T08 prompt versioning + gate** — engineering hygiene, invisible to stakeholders.
- **T09 sqlite provenance** — the in-session Audit-trail tab + saveable named sessions already demo provenance; persistence/retention/reproducibility is a deployment concern.
- **T10 source-credibility history** — T04's default-per-source-type weighting already demos the concept; per-source editing/history needs T09.
- **T11 multi-model cross-check** — 2× cost, needs two providers, marginal demo value.
- **T13 RAG** — the reach item; needs embeddings (§1), T09, and a populated approved corpus.
- **R-judge / R-cal / R-pair** — golden-set acceleration / calibration / prior-cancellation; all maturity features.

**Substitute in force now.** Single-call translation stays the demo default (cheap, full yes/partial/no relevance); the T06 structured pipeline is an optional advanced/auditable toggle; the slim in-session HITL (T12) provides the human-in-control workflow without sqlite/ensemble/cross-model dependencies.

**Re-introduction trigger.** Moving from POC to a paying or production engagement — when reproducible audit, prompt governance, calibrated confidence, or institutional memory become contractual requirements.

**Findings affected (NOT closed while parked).** **(6)** no prompt governance → T08; **(8)** no provenance → T09 (in-session audit is a partial stand-in); the measured-confidence half of **(3)** → T07. The span-grounding/injection half of **(2)/(3)** *is* delivered by T06.
