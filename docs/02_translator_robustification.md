# Translator Robustification: Plan to Move from Demo to Tool

## Executive Summary

The current headline-to-evidence translator (`src/translator.py`) is a single-shot LLM call that takes a 70-character headline, returns a posterior-shaped distribution over BN states, and injects that into pgmpy as virtual evidence. It works well enough for a podium demo but has eight named failure modes — semantic mismatch with the inference layer, headline-only input, no ensemble, no abstention, permissive validation, no prompt governance, no evaluation harness, no provenance — that together preclude using it as a decision-support tool.

This document proposes a 12-item programme across five categories — semantic foundations, input and reasoning, uncertainty quantification, governance and evaluation, operational integration — that closes those gaps. B1 splits into two execution slots (B1a/B1b) so the programme runs as 13 execution slots over 12 items. The plan is structured so that the first three items (A1: semantics, A2: schema hardening, D2: a 50-example golden set) compose a minimum viable correctness baseline that unblocks every subsequent measurement. Items B2 (span-grounded structured reasoning), B3 (abstention), and C1 (self-consistency ensemble) deliver the largest accuracy and calibration jump. Items D1, D3, E1, E2 build the institutional layer that converts a working tool into a governable one.

**Status legend.** ✅ = shipped. Nothing yet.

## Context

The translator sits between an analyst typing a headline into the dashboard and the BN engine running variable elimination. It has three responsibilities:

1. Decide which BN nodes the article speaks to.
2. For each such node, produce a soft distribution over that node's states.
3. Return a rationale a human can audit.

Output is consumed by [src/inference.py:123-139](src/inference.py#L123-L139), which wraps it in pgmpy `TabularCPD`s and uses pgmpy's virtual-evidence convention. The translator is provider-pluggable (Claude Code / OpenAI; see [src/translator.py:198-205](src/translator.py#L198-L205)) and is invoked once per headline from the dashboard.

The current implementation is ~510 lines: schema definition, system-prompt construction, two provider backends, a permissive validator, and a dispatcher. It has unit coverage of the validator but no coverage of translation quality.

This plan operates one layer below the existing roadmap in [docs/bn_app_next_steps.md](docs/bn_app_next_steps.md): the roadmap addresses how evidence *accumulates* and is *narrated*; this plan addresses how evidence is *produced*. The two documents share a downstream item — the news memory database E1 — which appears in both as a reach goal.

---

## Diagnosis: Why the Current Translator Is a Demo

The list below is the failure surface this plan closes. Items marked (M*/C*) are finding IDs from the master-plan §4 matrix and are included for completeness.

1. **Semantic mismatch with the inference layer (M2, C5).** The prompt at [src/translator.py:106-122](src/translator.py#L106-L122) asks for a *posterior-shaped* distribution that sums to 1. The inference layer at [src/inference.py:123-139](src/inference.py#L123-L139) consumes it as a *likelihood*. Every credible interval in the UI is in a slightly wrong place because of this single interface bug.
2. **Headline-only input.** A single sentence is the entire signal. The lede, body, source identity, dateline, and qualifiers ("no injuries," "unconfirmed," "Iranian state media reports") are dropped before the LLM ever sees them. The translator's accuracy ceiling is whatever a single sentence can convey.
3. **Single shot, zero ensemble.** One call at `temperature=0.0` per headline produces one answer. There is no self-consistency check, no multi-model cross-validation, and no notion of the translator's own confidence. `state_probs` is the LLM's hand-rolled posterior, not an empirically measured one.
4. **No abstention path.** The schema in [src/translator.py:75-131](src/translator.py#L75-L131) has no `relevant` flag. An off-topic headline still gets "validated 1 assignment" and silently enters inference. The only protection is the LLM's tendency to return an empty assignments list, which is neither enforced nor measured.
5. **Permissive validation that hides drift (C6, C7, C8).** Three accepted shapes for `state_probs` ([src/translator.py:235-289](src/translator.py#L235-L289)), silent renormalisation of any positive sum ([src/translator.py:288-289](src/translator.py#L288-L289) — `[0.99, 0.99, 0.99]` quietly becomes uniform), greedy-regex JSON extraction ([src/translator.py:297-312](src/translator.py#L297-L312)). Bad LLM output is indistinguishable from good in the audit trail.
6. **No prompt governance.** The system prompt is inlined Python regenerated at runtime from `STATES` ([src/translator.py:134-169](src/translator.py#L134-L169)). No version, owner, changelog, or eval gate. A prompt edit ships with no measurement and is invisible in git unless the reviewer reads the function body.
7. **No evaluation harness.** No golden set, no per-node F1, no calibration plot. "Does the translator work?" has no measurable answer.
8. **No provenance.** A translation lands in inference with the model name attached but no article URL, source credibility, prompt version, raw-response hash, or analyst-approval state. Reproducibility is best-effort.

---

## Category A: Semantic Foundations

These items fix the contracts between the translator and everything it touches. Nothing downstream is worth measuring until they are settled.

### A1. Likelihood semantics

**What exists now.** The system prompt at [src/translator.py:106-122](src/translator.py#L106-L122) instructs the LLM to produce a "probability distribution over states for this node" that "must sum to 1.0." This is a *posterior-shaped* output (`P(state | article)`). The inference layer at [src/inference.py:123-139](src/inference.py#L123-L139) wraps it in a single-column `TabularCPD` and uses pgmpy's `BeliefPropagation` virtual-evidence path, which interprets the values as *likelihoods* (`P(observation | state)`). The prior is therefore double-counted; see M2 in the review doc for a numerical example (a translator output of `{none:0.05, isolated:0.15, frequent:0.80}` combined with prior `P(frequent)≈0.10` produces a posterior near 60–70% on `frequent`, not 80%).

**What to add.** Pick a single semantics and align both ends to it. Three options:

1. *Likelihood-ratio output (recommended).* Prompt the LLM for relative evidence weights `ε_s = P(article | state=s) / max_s' P(article | state=s')`. This is a natural "likelihood" output with a clean reference point: the best-supported state has `ε=1.0` and others are fractions. It maps directly onto pgmpy's virtual-evidence convention without modification.
2. *Posterior output, prior-divided client-side.* Keep the current prompt; before injecting, compute `likelihood(s) = translator(s) / prior(s)` and renormalise. Requires that the engine expose the per-node prior at translation time and assumes the LLM's "prior" is the BN's prior — a non-trivial assumption.
3. *Posterior output, treated as evidence anyway (status quo, documented).* Accept the double-counting as a desirable damping effect and document it as a deliberate choice. Defensible only if the priors are intentionally "sticky" — this should be argued explicitly, not inherited.

**Why it matters.** This is the single most consequential change in the plan. Every credible interval, every robustness badge, every scenario percentage flows through this interface. Until it is settled, no downstream improvement to translator quality is measurable in the inference output.

**Recommendation.** Option 1. Add one paragraph to `docs/model_documentation.md` formalising the contract, update the system prompt, change `_validate_payload` to enforce `0 ≤ ε ≤ 1` with at least one `ε = 1`, and update `_virtual_evidence_cpds()` accordingly. Closes M2 and C5.

**Latent-regime impact.** Under the Plan 1 latent-regime topology (see `docs/01_latent_regime_plan.md`), the likelihood-ratio output of this item is exactly what feeds the Bayes-factor decomposition $\Lambda_{s_1, s_2} = P(E \mid S = s_1) / P(E \mid S = s_2)$. Option 1's $\varepsilon_s$ values are per-evidence-channel likelihoods over the emission node's states; Plan 1's engineering propagates them through the emission CPTs to produce regime-level Bayes factors. No additional work required in this plan — Option 1's contract is already the right shape.

### A2. Schema hardening

**What exists now.** `_validate_payload` at [src/translator.py:218-294](src/translator.py#L218-L294) accepts three shapes of `state_probs` (dict, list of `{state, prob}`, or JSON-encoded string), silently renormalises any positive sum, and the Claude Code path parses raw text through the greedy `\{.*\}` regex at [src/translator.py:297-312](src/translator.py#L297-L312). The OpenAI path is strict-mode JSON-schema constrained ([src/translator.py:341-349](src/translator.py#L341-L349)) but the Claude Code path is not.

**What to add.**

1. Single canonical shape — array of `{state, value}` — on both providers. Use `claude-agent-sdk`'s tool-use schema to make Claude Code's output strict-validated the same way OpenAI's `response_format` does.
2. Reject sums outside `[0.98, 1.02]` (under A1's likelihood semantics, reject any payload where no state has value `1.0`). Surface as a `TranslatorError` with the offending vector in the message.
3. Replace `_extract_json_block` with a brace-matching parser (or eliminate it entirely once Claude Code is schema-bound).
4. Snapshot the node taxonomy: include a hash of `STATES` in the prompt and reject if the LLM returns a node outside the snapshot. Surfaces silent drift between `network.py` edits and the prompt.

**Why it matters.** Closes C6, C7, C8. More importantly, removes silent-failure modes so that the evaluation harness in D2 actually measures what the model produces rather than what the validator silently coerced.

---

## Category B: Input and Reasoning

These items increase the signal available to the translator and the discipline of how it converts that signal into assignments.

### B1. Article-level input

> *Implementation note.* B1 splits into B1a (Article dataclass + paste-with-body + piped-feed inputs + default-per-source-type credibility) at execution slot 7, and B1b (per-source credibility editing with history) at slot 10. B1b sits after D3 (slot 9) because the audit log must be live before the credibility value at translation time can be pinned and recovered. See the execution-order table for the full sequencing.

**What exists now.** The dashboard passes a single headline string to `translate_headline()` at [src/translator.py:462-496](src/translator.py#L462-L496). There is no source, URL, body, dateline, or credibility metadata.

**What to add.** A new `Article` dataclass that carries `{headline, lede, body, source, source_type, url, published_at, language}` and a `translate_article()` entry point that takes it. Source type is a categorical: `wire_service`, `commercial_press`, `state_media`, `analyst_note`, `social_media`, `unknown`. The prompt receives the body (or first ~500 tokens of it) and is instructed to weight the body over the headline when they disagree.

For the demo workflow where an analyst pastes only a headline, the body is left blank and the prompt is told so explicitly. This degrades gracefully — the translator still works on a headline alone but knows it is working without full context.

**Why it matters.** Headlines are written for attention; bodies carry the qualifiers that disambiguate state. "Tanker attacked off Hormuz" could be `isolated` or `frequent` depending on whether the body says "the third such incident this week." Without the body, the translator is guessing.

**Source-credibility weighting.** Each article carries a scalar `w ∈ [0, 1]` that scales the magnitude of the likelihood ratios away from 1.0 (so a propaganda outlet gets less leverage than a Reuters wire on the same claim). Sources are addressed by stable identifier (domain, outlet name, or analyst-defined tag) and `w` is looked up at translation time from a user-editable table.

The table is a **living document** maintained from inside the dashboard:

- A "Sources" tab lists every source seen in the audit log alongside its current credibility score, the date of the last edit, and the analyst who edited it.
- The analyst can update an existing source's score or pre-populate a new source before its first article is translated.
- Every edit writes a new row to a `source_credibility_history` table — the score in force at translation time is the value most recently committed before that translation's `created_at` timestamp. Past translations are not retroactively rescored.
- A default fallback `w` per `source_type` covers unseen sources until the analyst assigns an explicit per-source value.

This makes the audit log fully reproducible (D3 stores the `(source_id, credibility_at_translation_time)` pair on every record) while letting the analyst's view of source quality evolve as new outlets appear and old ones change behaviour.

### B2. Structured reasoning with span-grounding

**What exists now.** Single LLM call: article in, JSON out. The model is free to invent claims that have no textual basis.

**What to add.** A three-step structured pipeline, each step a separate strict-schema LLM call:

1. **Claim extraction.** Input: article. Output: a list of atomic claims, each with `{subject, predicate, object, verbatim_span, confidence}`. `verbatim_span` is a copy-paste from the article body. Claims that don't appear as substrings of the article are rejected.
2. **Per-claim node mapping.** Input: a claim + the node taxonomy. Output: zero or one node assignment with `{node, state, likelihood_ratios, supporting_span_indices, reason}`. A claim that maps to no node is dropped silently.
3. **Per-node aggregation.** Input: all claim-level assignments. Output: one assignment per node, with likelihood ratios produced by combining per-claim ratios (multiplicative on log scale, renormalised).

The final output remains a list of `TranslatorAssignment` records, so downstream code is unaffected. The per-assignment audit log now carries the verbatim spans that supported it.

**Why it matters.** Eliminates hallucination at the assignment level — the rejection rule in step 1 forces every claim to be textually grounded, and the per-assignment supporting-span list is the audit surface an analyst needs to challenge a translation. Also makes the multi-claim-per-headline case ("Iran announced X; US responded Y") explicit instead of implicit.

**Cost note.** Three LLM calls per article rather than one. For demo cadence (≤10 articles/day) the latency is acceptable; for high-throughput ingest the steps can be merged into a single call with a stricter schema, at the cost of a weaker grounding guarantee.

### B3. Relevance filtering and abstention

**What exists now.** The system prompt nudges the LLM to "include only nodes the headline directly speaks to," but there is no enforced abstention path. An off-topic article still gets processed and may produce spurious assignments.

**What to add.** A `relevance` field on the top-level output: `{yes, partial, no}`. When `no`, the assignments list must be empty and the translation is logged but not injected into inference. When `partial`, assignments are accepted but flagged for analyst review in the HITL queue (E1).

A cheaper pre-filter runs before the expensive structured pipeline: a single embedding-distance check against a curated "topic anchor" document (descriptions of the Strait of Hormuz, US-Iran relations, regional energy markets). Below a threshold, the article is rejected as off-topic without an LLM call at all.

**Why it matters.** The translator currently spends compute on every input regardless of relevance. More importantly, it has no honest "I don't know" path — every article produces a confident-looking output. Abstention is what separates a tool from a demo.

---

## Category C: Uncertainty Quantification

These items convert the translator's confident point output into a measured distribution.

### C1. Self-consistency ensemble

**What exists now.** One LLM call at `temperature=0.0`. `state_probs` is whatever the model emits — a single number per state derived from no measurable process.

**What to add.** Replace the single call with N=5–10 calls at temperature ≈0.4, collect the resulting per-state likelihood vectors, and aggregate them into an empirical distribution. The aggregation is per-node:

- For each node that appears in any sample, compute the mean likelihood vector across samples in which it appears.
- Compute the sample-disagreement (e.g., mean total-variation distance from the aggregate) as a node-level uncertainty score.
- A node that appears in fewer than `M/2` of the N samples (where M = number of samples with any assignment) is dropped as "translator was inconsistent about whether to assign this at all."

The aggregated likelihood vector is what gets passed to inference. The disagreement score lives in the audit log and triggers analyst review (E1) above threshold.

**Why it matters.** This is the proper answer to "where does `state_probs` come from." Instead of trusting the LLM's gut-feel posterior, you measure the LLM's own consistency and let that drive the soft-evidence width. Disagreement across samples becomes honest uncertainty rather than synthetic confidence.

**Cost note.** N× LLM spend per article. At demo cadence this is negligible (~$0.05/article on Sonnet). At production cadence it scales linearly; below 1000 articles/day the bill is still small.

### C2. Multi-model cross-check

**What exists now.** The provider abstraction in [src/translator.py:462-496](src/translator.py#L462-L496) supports Claude Code and OpenAI but picks one. The other is dormant.

**What to add.** Run the self-consistency ensemble (C1) on both providers and compare the aggregates. Disagreement above threshold (e.g., per-node TV distance > 0.25) routes the article to the HITL queue (E1) with both translations attached. Agreement is the common case and ships through unchanged.

**Why it matters.** Self-consistency catches within-model uncertainty. Cross-model catches systematic biases (one model overweights tanker incidents, the other underweights militia activity). A two-model agreement is a much stronger signal than a single-model confident answer.

**Cost note.** Doubles the C1 cost. Optional toggle — recommend running on by default in the analyst workflow, off in batch ingest unless the article's confidence is borderline.

---

## Category D: Governance and Evaluation

These items make translator behaviour measurable, reviewable, and reproducible.

### D1. Prompt as versioned artefact

**What exists now.** The system prompt is inlined Python at [src/translator.py:134-169](src/translator.py#L134-L169), regenerated at runtime from the current `STATES`. Prompt edits ship without version, owner, or changelog.

**What to add.** Externalise the prompt to `prompts/translator/v{N}.yaml` with frontmatter `{version, owner, model, created, node_taxonomy_hash, changelog}`. The Python loader resolves the latest version unless pinned. The node-taxonomy block of the prompt body is auto-generated from `STATES` at load time and the hash recorded — any change to `network.py` that affects the taxonomy invalidates the prompt and requires a new version.

CI gate: any change to a prompt file requires the golden set (D2) to pass at ≥ baseline F1 minus a defined tolerance.

**Why it matters.** Prompt drift is the single most common cause of silent LLM-system regression. Externalising it puts the prompt under the same review discipline as code.

### D2. Golden set and continuous evaluation

**What exists now.** No labelled set, no metrics, no calibration data. `tests/` covers `_validate_payload` correctness but not translation quality.

**What to add.** A `tests/golden/translator/` directory of hand-labelled `(article, expected_output)` records. The v0 (slot 3 in the execution order) ships 30 records — enough to give A1/A2 a measurement loop. The size grows to 50 before D1's CI gate goes live (slot 8), so the gate has a defensible baseline. Each record covers every node at least once at v0 and at least twice by the 50-record mark, and includes edge cases (state media, hedged language, ambiguous applicability, off-topic). Target size after 6 months of operation: 300–500.

Metrics computed by a `pytest tests/test_translator_eval.py`:

- **Node-recall and node-precision.** Per node, did the translator assign when the gold did, and only then.
- **State accuracy conditional on node-match.** When the translator and gold agree on the node, do they agree on the top state?
- **Brier score on likelihood vectors.** Aggregate calibration metric on the soft outputs.
- **Calibration plots.** Bin predictions by confidence (under C1: empirical disagreement), measure realised accuracy per bin.
- **Abstention precision/recall.** Did the translator say `relevant=no` exactly when the gold did?

A small "translator accuracy" badge in the dashboard header surfaces the most recent run's headline numbers (node F1, calibration). Closes part of the "where do the numbers come from" trust gap.

**Why it matters.** Every prior item in this plan is unmeasurable without a golden set. D2 should be built early — even an MVP 30-record set provides a tighter feedback loop than no measurement at all.

**Implementation note.** Labelling cost is real once the workflow is in place. The golden set is the single most expensive artefact in this plan in human time.

### D3. Provenance and audit log

**What exists now.** `TranslatorResult` carries `{headline, assignments, rationale, model, provider, raw_response}` ([src/translator.py:51-67](src/translator.py#L51-L67)). The dashboard logs assignments into the observation list but does not persist source URL, prompt version, raw-response hash, or analyst-approval state.

**What to add.** Extend `TranslatorResult` to carry `{article_url, source, source_credibility, prompt_version, model, model_version, response_hash, temperature, ensemble_size, sample_disagreement, created_at, relevance, analyst_state, analyst_id, analyst_correction}`. Persist to a structured log (parquet or sqlite) keyed by `response_hash`. The dashboard observation log becomes a thin view over this store.

Reproducibility contract: given `(article, prompt_version, model, model_version, temperature, ensemble_size)`, re-running the translator should yield the same `response_hash` up to LLM sampling noise. Document the noise envelope.

**Why it matters.** Auditability is a hard requirement for any stakeholder-facing decision tool. It is also a prerequisite for E2 (RAG memory) — the retrieval corpus is exactly this log.

---

## Category E: Operational Integration

These items couple the translator to the wider workflow.

### E1. Human-in-the-loop review queue

**What exists now.** Every translation auto-commits as soft evidence. The analyst can edit it post-hoc in the dashboard, but the default flow has no checkpoint.

**What to add.** A confidence-driven queue. Translations land in one of three states:

- **Auto-approved.** Confidence above high threshold, no cross-model disagreement, source credibility above floor. Enters inference immediately.
- **Pending review.** Below high threshold or cross-model disagreement or source flagged. Visible in a "Triage" tab in the dashboard with both translations (if C2 was run), the article body, and the supporting spans (B2). Analyst approves, edits, or rejects.
- **Rejected.** Analyst declined. Logged but not used.

Edits are first-class records: the corrected output is the new ground truth and is fed back into the golden set queue (D2) for review.

**Why it matters.** Translator output cannot be the final arbiter on stakeholder-facing decisions. HITL is the bridge between an imperfect model and a defensible workflow. Coupling tightly with D2: today's analyst corrections are tomorrow's labelled examples.

### E2. Retrieval-augmented translation

**What exists now.** Each LLM call is stateless. Past translations of similar articles do not inform new ones.

**What to add.** Index the audit log (D3) by article embedding. For a new article, retrieve the top-K most similar past articles where `analyst_state ∈ {auto_approved, approved, edited}` and inject them as few-shot examples in the structured-reasoning prompt (B2). The injected examples are the analyst-approved final outputs, not the raw translator outputs.

**Why it matters.** Two compounding benefits: (a) the translator's behaviour stabilises around analyst-validated precedent rather than the LLM's untethered priors; (b) institutional knowledge accumulates in the corpus rather than evaporating after each session.

**Coupling.** E2 is the natural first consumer of E1 in [docs/bn_app_next_steps.md](docs/bn_app_next_steps.md). The two should share infrastructure — one news memory layer, two consumers (the translator for in-context examples, the narrative layer for cross-day synthesis).

**Implementation note.** This is the most architecturally ambitious item. It depends on D3 (provenance) and E1 (analyst approvals) being live and populated. Pursue last.

---

## Execution Order

The sequencing balances dependency chains, smallest-correctness-win-first, and stakeholder-visible impact. Each row lists the gap from the diagnosis it closes.

| Order | Item | Category | Closes | Rationale |
|-------|------|----------|--------|-----------|
| 1 | A1: Likelihood semantics | Foundations | (1) | The single most consequential interface bug. Until it is fixed, no downstream improvement is measurable in the inference output. Closes M2/C5 from the review. |
| 2 | A2: Schema hardening | Foundations | (5) | Removes silent-failure modes so the evaluation harness measures the model, not the validator. Closes C6/C7/C8. |
| 3 | D2 (MVP): Golden set v0 (30–50 records) | Governance | (7) | Without this, every subsequent change is unmeasurable. The most expensive item in human time; start it in parallel with A1/A2 so it is ready when the structured-reasoning work in B2 lands. |
| 4 | B3: Relevance filter and abstention | Reasoning | (4) | Cheap, high-impact. Adds an honest "not relevant" path before any other reasoning changes. The pre-filter (embedding check) saves cost on every off-topic input. |
| 5 | B2: Span-grounded structured reasoning | Reasoning | (2 partial, 5) | The largest single jump in translator trustworthiness. Eliminates hallucination at the assignment level by forcing every claim to cite verbatim source text. Depends on A2 (schema hardening) being clean. |
| 6 | C1: Self-consistency ensemble | Uncertainty | (3) | Replaces the LLM's hand-rolled `state_probs` with an empirically measured distribution. Depends on A2 being stable so the ensemble aggregator can rely on a canonical shape. |
| 7 | B1a: Article-level input | Reasoning | (2) | Article dataclass with all three input pathways supported (paste-only, paste-with-body, piped feed); default-per-source-type credibility weights only. Doubles the available signal without yet requiring the audit/pinning infrastructure. |
| 8 | D1: Prompt as versioned artefact | Governance | (6) | Operational hardening. Trivial to build once prompts are externalised; the value is the CI gate that prevents silent regressions. |
| 9 | D3: Provenance audit log | Governance | (8) | Reproducibility contract. Prerequisite for E2 (RAG retrieval needs a structured store), for B1b (per-source credibility pinning), and for any compliance review. |
| 10 | B1b: Per-source credibility with history | Reasoning | (2) | The Sources tab and `source_credibility_history` table. Depends on D3 (slot 9) being live so translations can be pinned to the credibility value in force at their `created_at`. |
| 11 | C2: Multi-model cross-check | Uncertainty | (3) | Compounds with C1 — catches systematic per-model biases. Optional toggle; default on for analyst workflow, off for batch ingest. |
| 12 | E1: HITL review queue (threshold-triggered) | Operations | (4, 7) | The bridge between an imperfect model and a defensible workflow. Threshold-triggered: only borderline translations enter the queue; the rest auto-approve. Compounds with D2: analyst corrections accrue into the golden set. |
| 13 | E2: Retrieval-augmented translation | Operations | (3, 6) | Reach item. Depends on D3 + E1 being populated. Couples directly to E1 in [docs/bn_app_next_steps.md](docs/bn_app_next_steps.md) — they should share one news-memory layer. |

**Minimum viable correctness baseline.** Items 1–3 (A1, A2, D2-MVP) fix the worst interface bugs and give the team a measurement loop. Everything after that compounds against that loop.

**Largest single quality jump.** Items 5–6 (B2, C1) together deliver the biggest stakeholder-visible improvement in translator output and calibration.

**Institutional infrastructure.** Items 8–13 build the layer that distinguishes a tool from a script: versioning, audit, review, memory.

---

## Design Decisions

Resolved decisions. One question remains open at the bottom of the section.

1. **Likelihood semantics (A1) — Decided: likelihood-ratio output.** The prompt asks the LLM for `ε_s = P(article | state=s) / max_s'`. The best-supported state pins at `ε=1.0`; others are fractions in `(0, 1]`. Maps directly onto pgmpy's virtual-evidence convention without modification. Closes M2/C5 from the review.
2. **Input pathway (B1) — Decided: all three supported, analyst chooses per article.** Paste-only headline (current behaviour preserved), paste-with-body, and a piped feed (RSS/GDELT) are all valid inputs. The `Article` dataclass tolerates missing `body`, `url`, `published_at`; the structured-reasoning prompt (B2) is told explicitly which fields are present so it can downgrade confidence when working from a headline alone.
3. **Golden-set authorship (D2) — Decided: single-author start, expand later.** Francesco labels the v0 set (30–50 records) alone. As resources expand, additional annotators are folded in; the doc schema already accommodates a per-record `annotator` field and inter-annotator agreement metrics can be added once N ≥ 2.
4. **Compute budget per article — Decided: accept demo-cadence cost for now.** Self-consistency at N=5 + 3-step structured reasoning multiplies per-article LLM cost ~15×. Acceptable at current demo cadence (≤10 articles/day). Revisit when daily volume passes 100 articles or when a high-throughput batch mode is proposed.
5. **Source-credibility table (B1) — Decided: living document, user-maintained from inside the dashboard.** The analyst assigns and updates per-source credibility scores `w ∈ [0, 1]` directly in a "Sources" tab. Defaults per `source_type` cover unseen sources. Every edit appends to a `source_credibility_history` table; past translations are pinned to the score in force at their `created_at` timestamp and are not retroactively rescored. The audit log (D3) carries `(source_id, credibility_at_translation_time)` on every record so reproducibility is preserved across edits.
6. **HITL operating model (E1) — Decided: threshold-triggered only.** The default flow auto-approves translations that clear the confidence and cross-model-agreement thresholds. Only borderline cases enter the analyst review queue. This is the lighter-weight UX shape and is consistent with the demo cadence in (4). Revisit if analyst workflow expands to always-on triage.
