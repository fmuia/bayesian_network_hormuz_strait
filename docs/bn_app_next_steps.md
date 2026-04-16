# BN Dashboard: Next Steps for Stakeholder Utility

## Executive Summary

The Streamlit dashboard is a functional prototype. This document proposes 11 improvements across five categories — inference mechanics, narrative/reporting, exploration, workflow, and knowledge infrastructure — to move it toward production use. The most critical finding: **the current evidence merging is last-observation-wins**, meaning the model has no memory across observations to the same node. Fixing this (A1) is the foundational prerequisite. After that, sensitivity attribution (A3) and pre-built scenario sequences (B2) provide the highest immediate value for committee demos and governance. A reach goal (E1) outlines a news memory database for institutional knowledge persistence.

## Context

The dashboard (`app/dashboard.py`) translates headlines into BN evidence, runs inference, displays scenario probabilities with credible intervals, and provides interactive network visualisation with manual override. Named sessions can be saved and restored.

---

## Category A: Inference and Evidence Mechanics

These improvements address how evidence enters, accumulates, and ages within the model — the computational foundation that everything else depends on.

### A1. Evidence Accumulation Across Observations

**What exists now.** When multiple headlines touch the same node across different days, the current merging logic is **last-observation-wins**: the most recent assignment to a node completely overwrites all previous ones (`_merged_evidence()` in `dashboard.py`, lines 402–413). Earlier observations are preserved in the log for display, but have zero effect on inference.

This means the model has no memory. If Day 1 reports "tanker harassed near Hormuz" (shifting `Tanker_Incidents` toward `isolated`) and Day 2 reports "second tanker incident this week" (shifting toward `frequent`), only Day 2's distribution survives. Worse, if Day 3 reports a diplomatic de-escalation headline that the translator maps back to `Tanker_Incidents ≈ none`, the two actual incidents from Days 1–2 vanish entirely from inference.

**What to add.** Replace last-observation-wins with a proper evidence accumulation mechanism. Two options, in order of sophistication:

*Option 1: Weighted pooling with recency decay.* Maintain all soft-evidence observations for each node. At inference time, combine them into a single distribution using time-weighted averaging — recent observations get higher weight, older ones decay toward the prior. The decay rate is a user-adjustable parameter. This handles both the accumulation problem and the staleness problem in one mechanism.

Concretely, for a node with observations at days $t_1, \ldots, t_n$ producing distributions $d_1, \ldots, d_n$, the merged distribution is:

$$d_{\text{merged}}(s) = \frac{\sum_i w_i \cdot d_i(s)}{\sum_i w_i}, \quad w_i = \exp(-\lambda \cdot (t_{\text{now}} - t_i))$$

where $\lambda$ controls the decay rate. With $\lambda = 0$ this is uniform averaging (all observations equally weighted); with large $\lambda$ it approximates last-observation-wins.

*Option 2: Sequential Bayesian updating.* Treat each headline's soft evidence as a likelihood and multiply sequentially into the node's distribution, starting from the prior. Each observation tightens or shifts the distribution rather than replacing it. This is the textbook Bayesian approach but requires care: the translator's output is a posterior-like distribution (sums to 1), not a pure likelihood ratio. The translator prompt or a post-processing step would need to convert the output into a form suitable for multiplicative combination.

**A design choice within Option 1:** Weighted averaging of probability vectors is not the only pooling method. It produces spread-out distributions that can be diffuse when observations disagree. An alternative is *logarithmic pooling* (geometric mean, then renormalise), which produces sharper distributions peaked where observations agree — often more appropriate for combining independent evidence. The choice between arithmetic and logarithmic pooling should be treated as a design parameter, tested against representative headline sequences, and documented.

**Recommendation:** Start with Option 1 (weighted pooling with arithmetic averaging). It is robust, interpretable, easy to explain to stakeholders ("recent evidence counts more"), and does not require changes to the translator. Option 2 is the theoretically correct extension for later. Within Option 1, the arithmetic-vs-logarithmic pooling choice can be evaluated empirically once the infrastructure is in place.

**Supporting UI changes:** Visual staleness indicators on observed nodes (muted/hatched when old). Per-node evidence history panel showing the full stack of observations that contributed to the merged distribution.

### A2. Node-Level Credible Intervals

**What exists now.** Credible intervals (Dirichlet resampling, m=200, concentration=20) are computed for the terminal Scenario node only.

**What to add.** Extend `scenario_credible_intervals` (or write a parallel function) to compute credible intervals for every node's posterior marginal, not just Scenario. Display these as error bars on the posterior bar charts in the node detail panel (right side of the Network tab).

**Why it matters.** When an expert clicks on `Strait_Operationally_Closed` and sees "partial: 52%", they should also see the uncertainty band. Is it 52% +/- 3pp (robust) or 52% +/- 15pp (fragile)? This identifies which parts of the causal chain are stable conclusions and which are sensitive to CPT specification.

### A3. Sensitivity Attribution — "What Drove the Change?"

**What exists now.** When scenario probabilities shift after a new headline, the user can see the new numbers and inspect the observation log, but there is no accounting of *which node assignment was responsible for how much of the probability change*.

**What to add.** After each new observation, compute and display a probability-change waterfall: for each node that was set or updated, show its marginal contribution to the shift in scenario probabilities (leave-one-out decomposition). Display as a horizontal bar chart: "Tanker\_Incidents → frequent contributed +12pp to Severe Closure; Diplomatic\_Resolution\_Path → narrowing contributed +8pp."

**Why it matters.** This is the single most important feature for governance. When probabilities change, the committee needs to know *why*. Automated attribution makes the model's reasoning transparent and debatable. It also surfaces CPT sensitivities organically: if a single node causes a disproportionate swing, that's either a genuine high-leverage variable or a CPT that needs revisiting.

**Computational cost note.** Leave-one-out decomposition requires N+1 inference calls (once with all evidence, once per observed node removed). On the base CPTs (without Dirichlet resampling) this is fast — Variable Elimination is sub-second. But if combined with the full Dirichlet resampling (m=200 draws per call), the cost multiplies to (N+1) × 200 inference runs, which may take several seconds with 8+ observed nodes. The recommended approach: compute the attribution waterfall on point-estimate CPTs for real-time display, and offer the full Dirichlet-resampled version as an optional deeper analysis accessible via an expander or button.

---

## Category B: Narrative, Timeline, and Reporting

These improvements address how the model's outputs are communicated to stakeholders — both during sessions and in post-meeting artefacts.

### B1. Daily Narrative Generation

**What exists now.** Each headline gets a translator rationale (one or two sentences from the LLM explaining the node assignments). There is no synthesis across multiple headlines within a day, and no narrative connecting one day's evidence to the next.

**What to add.** Two layers of LLM-generated narrative:

*Daily summary.* At the end of each day (or on demand), the app generates a structured narrative that synthesises all observations entered that day into a coherent paragraph. This includes: which nodes were affected, in which direction, and what the net effect on scenario probabilities was. The summary is tied to a visual timeline — a vertical axis of days, with each day showing its headline(s), the narrative summary, and a compact sparkline or delta indicator for each scenario probability.

The daily summary prompt would receive: the day's headlines, the translator assignments, the before/after scenario probabilities, and the attribution waterfall (from A3). The LLM's job is to weave these into a readable paragraph: "Day 4 saw two escalation signals. The tanker incident report shifted Tanker\_Incidents toward 'frequent,' while the military deployment headline elevated US\_Military\_Response. Together, these moved Severe Closure from 18% to 31%, with tanker incidents contributing the larger share. Diplomatic channels remain unobserved."

*Monthly meta-narrative for stakeholder meetings.* A separate synthesis that collapses the daily summaries into a month-level briefing. This summarises the overall trajectory ("Over the past month, scenario probabilities shifted steadily toward Prolonged Conflict, driven primarily by three clusters of evidence: ..."), identifies the key turning points ("The largest single-day shift occurred on Day 12 when ..."), and highlights unresolved uncertainties ("No evidence has been observed on Third\_Party\_Mediation or Iranian\_Regime\_Stability; these nodes remain at their priors").

The monthly narrative is generated from the full observation log and probability evolution data. It should be exportable as part of the session report (B3).

**Timeline visualisation.** The probability evolution chart already shows scenario probabilities by day. Extend it with an annotated timeline: each day gets a clickable marker showing the headline(s) and the daily narrative. Hovering or clicking reveals the full summary. This transforms the probability evolution chart from a bare line graph into a narrated story of how the situation developed.

### B2. Pre-Built Scenario Sequences

**What exists now.** Six example headlines in `src/evidence.py` that can be clicked one at a time. They are isolated — there is no notion of a coherent multi-day narrative.

**What to add.** Curated scenario sequences: ordered lists of 4–6 headlines that tell a coherent escalation or de-escalation story across multiple days:

- **Rapid escalation**: Day 1: "IRGC announces inspection regime on Hormuz traffic" → Day 2: "Fourth tanker incident in two weeks" → Day 3: "US conducts strikes against IRGC naval assets" → Day 4: "Major fire at Ras Tanura terminal; strait closed"
- **Diplomatic resolution**: Day 1: "IRGC announces inspection regime" → Day 2: "Oman confirms active US–Iran back-channel talks" → Day 3: "Treasury issues 90-day sanctions waiver" → Day 4: "Iran and US agree to interim maritime safety protocol"
- **Stalemate**: A mixed sequence where escalation and de-escalation signals alternate, producing sustained uncertainty.

A "Play sequence" button advances through the headlines automatically, day by day, showing how the probability evolution chart and daily narratives build up.

**Why it matters.** Live demonstrations are how models gain institutional trust. Running a curated sequence in a committee meeting takes 2 minutes and shows the model reacting coherently to a plausible narrative arc. Pre-built sequences let the presenter choose a narrative matching the committee's current concern.

### B3. Session Export and Reporting

**What exists now.** Sessions can be saved and loaded within the app (JSON store in `data/`). There is no export mechanism for sharing results with people who are not running the dashboard.

**What to add.** An "Export" button that generates a self-contained report:

- Current scenario probabilities with credible intervals.
- The probability evolution chart as a static image.
- The annotated timeline with daily narratives (from B1).
- The full observation log with day, headline, node assignments, and translator rationale.
- The network diagram snapshot (the Graphviz render already exists in `src/viz.py`).
- The monthly meta-narrative (from B1), if available.
- A metadata header: date, number of observations, translator provider and model, session name.

Output as a PDF or standalone HTML file. The observation log and audit trail provide the governance record; the narratives and charts provide the executive summary.

---

## Category C: Exploration and Analysis

These improvements support interactive exploration of the model — what-if analysis, scenario comparison, and model introspection.

### C1. Scenario Comparison Mode

**What exists now.** The app tracks a single evidence thread. There is no way to compare two different assumptions side-by-side.

**What to add.** A "Compare" mode that forks the current session into two parallel branches — e.g., "Diplomacy succeeds" vs "Diplomacy fails" — and displays scenario probabilities for each branch side by side. Implementation: duplicate the current merged evidence into two copies, let the user add different observations to each branch, run inference on both, render in a two-column layout.

**Why it matters.** Investment committees think in terms of "if X happens, what follows?" Comparison mode lets them hold two futures in view simultaneously, which is how scenario planning actually works. It also makes the BN's conditional structure tangible: flipping a single node visibly shifts probabilities.

### C2. CPT Exploration and Elicitation Support

**What exists now.** CPTs are hardcoded in `src/network.py`. The dashboard displays posterior marginals but does not expose the CPTs themselves. An expert who wants to inspect or challenge a specific probability must read source code.

**What to add.** A new tab — "Model internals" or "CPT explorer" — that displays each node's CPT as an interactive table:

- Parent states as row/column headers, conditional probabilities as cell values.
- Highlight the column corresponding to the current evidence configuration.
- Allow the user to temporarily adjust a CPT entry and see the effect on scenario probabilities in real time. Changes are session-local and do not persist to code.
- Optionally, per-column sensitivity analysis showing how much the scenario posterior would change if that column's probabilities shifted within a plausible range.

**Why it matters.** This is the elicitation support tool described in `bn_hmm_integration.md` (Section 4.3). The iterative elicitation workflow — elicit, run inference, show experts the implications, adjust — requires a UI that makes CPTs visible and editable without touching code. It also addresses the common objection "I don't know where these numbers came from."

**Downstream relevance for HMM integration.** The CPT explorer is also the natural place to design and validate the BN-to-HMM mappings described in `bn_hmm_integration.md`: translating the BN's discrete node states (e.g., `Oil_Price_Regime` ∈ {`below_90`, `90_to_120`, `above_120`}) into continuous priors on HMM emission parameters (Section 3.1.5) or into regression coefficients for emission modification (Section 3.3). These mappings involve non-trivial design choices that benefit from interactive exploration.

### C3. Observation Undo/Redo and Evidence Pinning

**What exists now.** Observations can be removed individually. There is no undo mechanism and no way to protect specific observations from being cleared.

**What to add.**

1. **Undo/redo stack.** Each observation addition or removal is pushed to an undo stack. A button or shortcut reverses the last action.
2. **Evidence pinning.** An observation can be "pinned," meaning it survives a session reset and cannot be accidentally removed. Useful for anchoring known facts while exploring hypotheticals.

**Why it matters.** Undo/redo removes the fear of making mistakes, which is the primary barrier to exploratory use. Pinning lets the user separate facts from hypotheses.

---

## Category D: Workflow and Scale

These improvements support the daily operational workflow — handling multiple headlines efficiently and evolving toward a production monitoring system.

### D1. Multi-Headline Batch Processing

**What exists now.** Headlines are entered one at a time, each triggering a separate LLM call.

**What to add.** A "Batch" input mode that accepts multiple headlines (one per line or pasted from a news feed), translates them all, and enters them as observations on the current day. Show a preview of the combined effect before committing: "These 4 headlines together would shift Severe Closure from 18% to 34%. Commit all / select individually?"

**Why it matters.** In a real monitoring workflow, the analyst has a morning stack of 5–10 relevant headlines. Batch processing reflects this reality. The preview step is critical because individual headlines can produce contradictory signals.

---

## Category E: Reach Goals — Knowledge Infrastructure

These are longer-horizon improvements that build institutional memory and connect the BN to a persistent information layer.

### E1. News Memory Database and Ontology

**What exists now.** No persistent store of processed headlines. Each is forgotten by inference once overwritten. No deduplication, cross-session search, or pattern analysis.

**What to add.** A structured news memory layer (knowledge graph or vector store, e.g., **Cognee**) that persists every headline alongside its translator output and session metadata. Key capabilities:

- **Ontology-aligned storage:** Headlines indexed against the BN's node vocabulary (`STATES` dict), enabling queries like "all headlines touching `Tanker_Incidents` in the last 90 days."
- **Deduplication and contradiction detection:** Flag semantically similar past headlines; detect when a new headline contradicts recent evidence.
- **Pattern recognition:** Meta-analysis of which nodes are frequently/never observed, translator calibration biases.
- **Retrieval-augmented translation:** Use similar past headlines as LLM context for consistency.

The BN's node vocabulary is already a lightweight ontology; the memory should be indexed against it so queries operate in the same conceptual space as inference.

**Implementation note.** This is the most architecturally ambitious step. It depends on A1 (evidence accumulation) and B1 (narrative generation) and should be pursued only after the core layers are stable.

---

## Execution Order

The table below suggests a sequencing that balances foundational correctness, stakeholder impact, and dependency chains.

| Order | Item | Category | Rationale |
|-------|------|----------|-----------|
| 1 | A1: Evidence accumulation | Inference | Foundational fix. The model currently has no memory across observations to the same node. Every downstream improvement assumes this works. |
| 2 | A3: Sensitivity attribution | Inference | High governance value. Low implementation cost (leave-one-out over existing inference). Immediately useful in committee settings. |
| 3 | B2: Pre-built scenario sequences | Narrative | High demo value. Low effort (data only, no new inference logic). Makes the app presentable in stakeholder meetings. |
| 4 | B1: Daily narrative generation | Narrative | Enhanced by A3 (attribution enriches the narrative, but B1 can function without it using raw probability deltas). Transforms the app from a dashboard into a briefing tool. The monthly meta-narrative layer follows naturally. |
| 5 | C1: Scenario comparison mode | Exploration | High analytical value. Moderate effort (parallel evidence threads, dual rendering). Natural complement to B2 sequences. |
| 6 | B3: Session export | Narrative | Depends on B1 (narratives are the most valuable export content). Delivers the governance artefact: a distributable PDF with full audit trail. |
| 7 | A2: Node-level credible intervals | Inference | Extends existing Dirichlet resampling. Moderate compute cost. Valuable for CPT refinement and elicitation workflows. |
| 8 | C2: CPT explorer | Exploration | Depends on A2 (per-column sensitivity is the key feature). Enables the iterative elicitation loop described in `bn_hmm_integration.md`. |
| 9 | D1: Batch processing | Workflow | Quality-of-life for daily operations. Independent of other improvements but more useful once A1 (accumulation) exists. |
| 10 | C3: Undo/redo and pinning | Exploration | Quality-of-life for interactive use. Independent but compounds with C1 (comparison mode). |
| 11 | E1: News memory database | Knowledge | Reach goal. Depends on A1, B1, and stable translator behaviour. Architecturally ambitious; pursue once the core layers are proven. |
