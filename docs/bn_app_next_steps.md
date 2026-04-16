# BN Dashboard: Next Steps for Stakeholder Utility

## Context

The Streamlit dashboard (`app/dashboard.py`) is a functional prototype: it translates headlines into BN evidence, runs inference, displays scenario probabilities with credible intervals, and provides an interactive network visualisation with manual override. Named sessions can be saved and restored.

This document proposes incremental improvements that move the app from a demonstration tool toward something stakeholders would use in structured deliberation. Improvements are grouped by category, and an execution-order table at the end provides a suggested sequencing.

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

**Recommendation:** Start with Option 1 (weighted pooling). It is robust, interpretable, easy to explain to stakeholders ("recent evidence counts more"), and does not require changes to the translator. Option 2 is the theoretically correct extension for later.

**Supporting UI changes:** Visual staleness indicators on observed nodes (muted/hatched when old). Per-node evidence history panel showing the full stack of observations that contributed to the merged distribution.

### A2. Node-Level Credible Intervals

**What exists now.** Credible intervals (Dirichlet resampling, m=200, concentration=20) are computed for the terminal Scenario node only.

**What to add.** Extend `scenario_credible_intervals` (or write a parallel function) to compute credible intervals for every node's posterior marginal, not just Scenario. Display these as error bars on the posterior bar charts in the node detail panel (right side of the Network tab).

**Why it matters.** When an expert clicks on `Strait_Operationally_Closed` and sees "partial: 52%", they should also see the uncertainty band. Is it 52% +/- 3pp (robust) or 52% +/- 15pp (fragile)? This identifies which parts of the causal chain are stable conclusions and which are sensitive to CPT specification.

### A3. Sensitivity Attribution — "What Drove the Change?"

**What exists now.** When scenario probabilities shift after a new headline, the user can see the new numbers and inspect the observation log, but there is no accounting of *which node assignment was responsible for how much of the probability change*.

**What to add.** After each new observation, compute and display a probability-change waterfall: for each node that was set or updated, show its marginal contribution to the shift in scenario probabilities (leave-one-out decomposition). Display as a horizontal bar chart: "Tanker\_Incidents → frequent contributed +12pp to Severe Closure; Diplomatic\_Resolution\_Path → narrowing contributed +8pp."

**Why it matters.** This is the single most important feature for governance. When probabilities change, the committee needs to know *why*. Automated attribution makes the model's reasoning transparent and debatable. It also surfaces CPT sensitivities organically: if a single node causes a disproportionate swing, that's either a genuine high-leverage variable or a CPT that needs revisiting.

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

**What exists now.** Each headline is translated, observed, and forgotten (by inference) once a newer observation overwrites it. There is no persistent store of processed headlines, no deduplication, and no way to search past evidence or identify patterns across sessions.

**What to add.** A structured news memory layer — a knowledge graph or vector store that persists every headline alongside its translator output (node assignments, confidence distributions, rationale) and session metadata (day, session name, scenario probabilities at time of ingestion).

A tool like **Cognee** (or a lighter-weight combination of a vector database with structured metadata) would provide:

1. **Ontology-aligned storage.** Each headline is stored with its mapping to the BN's node vocabulary. This creates a queryable corpus organised by the same causal structure as the model: "show me all headlines that touched `Tanker_Incidents` in the last 90 days" or "what evidence has ever been observed for `Iranian_Regime_Stability = unstable`?"

2. **Deduplication and contradiction detection.** When a new headline arrives, the memory layer checks whether a semantically similar headline has already been processed. If so, it flags the overlap and optionally reuses the previous translation rather than making a fresh LLM call. It can also detect contradictions: "This headline implies `Negotiations = success`, but 2 days ago a headline implied `Negotiations = breakdown` — flag for analyst review."

3. **Pattern recognition across sessions.** Over time, the memory accumulates a corpus of headline → BN mapping pairs. This enables meta-analysis: which nodes are most frequently observed? Which nodes are never touched by headlines (suggesting either a gap in news coverage or a node that the translator struggles to map to)? Are there systematic biases in the translator's confidence calibration?

4. **Retrieval-augmented translation.** When translating a new headline, the system retrieves the most similar past headlines and their translations as context for the LLM. This improves consistency: if "tanker incident in Hormuz" was mapped to `Tanker_Incidents = isolated` last month, a very similar headline today should produce a similar mapping unless the context has materially changed.

**Ontology considerations.** The BN's node vocabulary (`STATES` dict in `src/network.py`) is already a lightweight ontology — 13 nodes with named states covering the causal space. The news memory should be indexed against this ontology so that queries and retrieval operate in the same conceptual space as inference. Cognee's graph-based approach is a natural fit here: entities are BN nodes, relationships are BN edges, and observations are facts attached to entity-states with timestamps.

**Why it matters.** As the system moves from demo toward production, institutional memory becomes essential. An analyst starting a new session should be able to ask "what did we observe last month?" and get a structured answer, not scroll through a JSON file. The knowledge graph also supports the monthly meta-narrative (B1): the LLM generating the monthly summary can query the memory for the full evidence history rather than relying only on the current session's observation log.

**Implementation note.** This is the most architecturally ambitious step and depends on several earlier improvements (particularly A1 evidence accumulation and B1 narrative generation). It should be pursued only after the core inference and narrative layers are stable.

---

## Execution Order

The table below suggests a sequencing that balances foundational correctness, stakeholder impact, and dependency chains.

| Order | Item | Category | Rationale |
|-------|------|----------|-----------|
| 1 | A1: Evidence accumulation | Inference | Foundational fix. The model currently has no memory across observations to the same node. Every downstream improvement assumes this works. |
| 2 | A3: Sensitivity attribution | Inference | High governance value. Low implementation cost (leave-one-out over existing inference). Immediately useful in committee settings. |
| 3 | B2: Pre-built scenario sequences | Narrative | High demo value. Low effort (data only, no new inference logic). Makes the app presentable in stakeholder meetings. |
| 4 | B1: Daily narrative generation | Narrative | Depends on A3 (attribution feeds the narrative). Transforms the app from a dashboard into a briefing tool. The monthly meta-narrative layer follows naturally. |
| 5 | C1: Scenario comparison mode | Exploration | High analytical value. Moderate effort (parallel evidence threads, dual rendering). Natural complement to B2 sequences. |
| 6 | B3: Session export | Narrative | Depends on B1 (narratives are the most valuable export content). Delivers the governance artefact: a distributable PDF with full audit trail. |
| 7 | A2: Node-level credible intervals | Inference | Extends existing Dirichlet resampling. Moderate compute cost. Valuable for CPT refinement and elicitation workflows. |
| 8 | C2: CPT explorer | Exploration | Depends on A2 (per-column sensitivity is the key feature). Enables the iterative elicitation loop described in `bn_hmm_integration.md`. |
| 9 | D1: Batch processing | Workflow | Quality-of-life for daily operations. Independent of other improvements but more useful once A1 (accumulation) exists. |
| 10 | C3: Undo/redo and pinning | Exploration | Quality-of-life for interactive use. Independent but compounds with C1 (comparison mode). |
| 11 | E1: News memory database | Knowledge | Reach goal. Depends on A1, B1, and stable translator behaviour. Architecturally ambitious; pursue once the core layers are proven. |
