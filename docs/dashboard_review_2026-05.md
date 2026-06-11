# Dashboard Review — Explorations Branch (2026-05)

**Purpose.** Snapshot audit of the current dashboard across math/construction, code implementation, and visualization. Authored as the working document for the `explorations` branch, intended to be walked through systematically. Each finding is self-contained: file/line refs, what's wrong, why it matters, suggested direction.

**Scope.** Everything currently merged into the branch as of 2026-05-23 (after A2 — node-level credible intervals — shipped in `dashboard_refinements`). The roadmap in `bn_app_next_steps.md` operates at the feature level; this review operates *below* the roadmap, at the level of hidden semantic choices, correctness, and polish in the existing code.

## Status legend

- **Open** — not yet investigated or actioned.
- **Investigating** — under review; no decision yet.
- **Accepted** — decision made to address; tracked but not yet implemented.
- **Implemented** — fix shipped (link the commit/PR).
- **Wontfix** — reviewed and consciously declined; capture the reasoning inline.

## Severity legend

- 🔴 **P0** — Correctness or interpretation. The model or UI says one thing and means another, or behavior degrades under realistic use (multi-user, long sessions).
- 🟡 **P1** — Quality, performance, or material UX issue. Doesn't break correctness but constrains future work or harms stakeholder trust.
- 🟢 **P2** — Polish, dead code, minor inconsistency.

---

## Findings Index

| ID | Severity | Title | Status |
|----|----------|-------|--------|
| **Math / Construction** | | | |
| M1 | 🔴 P0 | Scenario node is a softmax-classifier of three parents, not a probabilistic outcome | Open |
| M2 | 🔴 P0 | Soft-evidence semantics: translator output treated as likelihood, not posterior belief | Open |
| M3 | 🟡 P1 | Dirichlet concentration κ=20 is uniform across every CPT | Open |
| M4 | 🟡 P1 | Within-CPT columns sampled independently — no correlated shape uncertainty | Open |
| M5 | 🟡 P1 | DAG omissions called out in UI are real channels, not pedantic | Open |
| M6 | 🟡 P1 | Root priors are unjustified and propagate everywhere | Open |
| M7 | 🟡 P1 | Resample-mean vs point-estimate posterior used inconsistently in UI | Open |
| M8 | 🟢 P2 | `+1e-6` Dirichlet alpha guard never fires meaningfully | Open |
| M9 | 🟢 P2 | `scenario_credible_intervals` and `node_credible_intervals` duplicate the resampling loop | Open |
| **Code Implementation** | | | |
| C1 | 🔴 P0 | `engine` is `@st.cache_resource` but is mutated on every rerun | Open |
| C2 | 🔴 P0 | `dashboard.py` is 1878 lines / 76 KB in a single file | Open |
| C3 | 🟡 P1 | Probability-evolution chart recomputes full posterior series on every rerun | Open |
| C4 | 🟡 P1 | Streamlit caches have no bounds | Open |
| C5 | 🟡 P1 | Soft evidence as virtual-evidence-CPD encodes a likelihood mismatch (mirror of M2) | Open |
| C6 | 🟡 P1 | Translator validation accepts dict / list / string forms of `state_probs` | Open |
| C7 | 🟡 P1 | `_validate_payload` re-normalizes any positive-sum probability vector silently | Open |
| C8 | 🟡 P1 | `_extract_json_block` regex is fragile | Open |
| C9 | 🟡 P1 | `render_network_png` is dead code | Open |
| C10 | 🟡 P1 | `_NODE_LEVEL` in `viz.py` is a manually maintained DAG layout | Open |
| C11 | 🟢 P2 | No test for `_merged_evidence` soft↔hard ordering invariants | Open |
| C12 | 🟢 P2 | `_PLUGINS_REGISTERED` workaround needs a TODO | Open |
| C13 | 🟢 P2 | `Observation.tone` is declared but unused | Open |
| C14 | 🟢 P2 | No tests for `viz.py` or dashboard helpers | Open |
| **Visualization** | | | |
| V1 | 🔴 P0 | Headline scenario card numbers are resample-mean, not point-estimate | Open |
| V2 | 🟡 P1 | Information density is wrong: three giant cards for three numbers | Open |
| V3 | 🟡 P1 | Robustness thresholds (±8, ±20 pp) are arbitrary; sharp emoji transitions | Open |
| V4 | 🟡 P1 | Manual override sliders are a UX trap (must sum to exactly 100) | Open |
| V5 | 🟡 P1 | Hard-observed nodes show an uninformative bar chart | Open |
| V6 | 🟡 P1 | DAG layout: level 3 has four stacked nodes, level 4 has only Scenario | Open |
| V7 | 🟡 P1 | Evolution chart conflates parameter uncertainty with forecast uncertainty | Open |
| V8 | 🟡 P1 | ~350 lines of CSS embedded inline | Open |
| V9 | 🟢 P2 | No before/after visualization on a new observation | Open |
| V10 | 🟢 P2 | Latest-translation panel renders distributions as plain text | Open |
| V11 | 🟢 P2 | Green/red distinction is colorblind-hostile | Open |
| V12 | 🟢 P2 | Headline truncation on evolution-chart tooltip is aggressive (180 chars) | Open |
| V13 | 🟢 P2 | Edge-rationale tab is isolated from the graph | Open |
| V14 | 🟢 P2 | Scenario cards have a fixed narrative paragraph that doesn't change with evidence | Open |
| V15 | 🟢 P2 | 460px DAG canvas height is calibrated to the right column — fragile coupling | Open |

---

## Math / Construction

### M1. Scenario node is a softmax-classifier of three parents, not a probabilistic outcome

**Severity.** 🔴 P0
**Files.** `src/network.py:326-361`
**Status.** Open

#### Setup

Name the relevant nodes:

- $D$ = Energy_Infrastructure_Damage $\in$ {none, moderate, severe}
- $T$ = Conflict_Duration $\in$ {short, medium, long}
- $P$ = Diplomatic_Resolution_Path $\in$ {open, narrowing, closed}
- $S$ = Scenario $\in$ {Stress_Mitigates, Prolonged_Conflict, Severe_Closure}

The DAG has only $D, T, P \to S$. No other arrows enter $S$, and $S$ is never observed (the translator explicitly skips it). All evidence $E$ is on upstream nodes.

#### What variable elimination actually computes

Because $S$ is a leaf with parent set $\text{Pa}(S) = \{D, T, P\}$, and $E$ is upstream of $\text{Pa}(S)$:

$$
P(S = s \mid E) \;=\; \sum_{d,t,p} P(S = s \mid D = d, T = t, P = p) \cdot P(D = d, T = t, P = p \mid E)
$$

Define:

- $f(s \mid d, t, p) := P(S = s \mid D = d, T = t, P = p)$ — the elicited CPT (a 27-column lookup).
- $\pi(d, t, p \mid E) := P(D = d, T = t, P = p \mid E)$ — the joint posterior over the three parents.

Then:

$$
P(S = s \mid E) \;=\; \mathbb{E}_{\pi(\cdot \mid E)} \big[\, f(s \mid D, T, P) \,\big]
$$

**In words:** the scenario posterior is the expected value of a fixed 3-class classifier $f$, taken under the joint posterior of its three inputs. Structurally identical to applying a softmax classifier to a soft input — the BN is producing $\pi$; $f$ is a labeling layer on top.

#### Quantifying how "classifier-like" $f$ is

Column-by-column entropy of $f$ (max at 3 states is $\log_2 3 \approx 1.585$ bits):

| $(D, T, P)$                  | $f$ values            | $H(f)$ (bits) |
| ---------------------------- | --------------------- | ------------- |
| (none, short, open)          | $[0.94, 0.05, 0.01]$  | 0.36          |
| (severe, long, closed)       | $[0.01, 0.09, 0.90]$  | 0.55          |
| (moderate, medium, narrowing)| $[0.20, 0.60, 0.20]$  | 1.37          |

Corner columns are near-deterministic; interior columns (where the parents disagree) are softer. So $f$ is mostly a hard classifier with soft regions where the parents conflict.

#### Consequence 1 — no information added by $S$

The Scenario posterior contains exactly the information already present in $\pi(d, t, p \mid E)$, run through a known deterministic function $f$. Equivalently:

$$
\text{reporting } \pi(d, t, p \mid E) \quad \Longleftrightarrow \quad \text{reporting } P(S \mid E) \text{ under any chosen classifier } f
$$

If a stakeholder pushes back on a scenario number, the disagreement is necessarily one of:

1. The joint posterior $\pi$ — i.e. an objection to upstream CPTs or evidence.
2. The labeling function $f$ — i.e. an objection to how $(D, T, P)$ outcomes get mapped to scenario names.

These are different kinds of disagreement and deserve different UI surfaces.

#### Consequence 2 — credible intervals on $S$ are bounded by those on $\pi$

With independent Dirichlet resampling at concentration $\kappa$, the variance of a single CPT entry $\theta$ is

$$
\text{Var}[\theta] \;\approx\; \frac{\theta(1-\theta)}{\kappa + 1}
$$

For a corner entry $\theta = 0.94$, $\kappa = 20$: stddev $\approx 5.2$ pp. For a middle entry $\theta = 0.60$: stddev $\approx 10.7$ pp.

The Scenario-CPT contribution to $\text{Var}[P(S = s \mid E)]$ is approximately

$$
\sum_{d,t,p} \pi(d, t, p \mid E)^{2} \cdot \text{Var}[\theta_{s \mid dtp}]
$$

Because $\pi$ typically spreads mass across several parent configurations, $\sum \pi^2 \ll 1$, so the Scenario-CPT contribution is small.

**The dominant share of the CI on $S$ comes from upstream Dirichlet resampling perturbing $\pi$ itself**, not from resampling $f$. Verifiable: hold the Scenario CPT fixed across resamples and the CIs barely move.

#### Consequence 3 — the implicit conditional-independence claim

By d-separation, the model asserts:

$$
S \,\perp\!\!\!\perp\, \{\text{Tankers}, \text{Military}, \text{Strait}, \text{Militia}, \text{Sanctions}, \dots\} \;\big|\; \{D, T, P\}
$$

Operationally: given the level of damage, duration, and diplomatic path, the *causes* that produced those outcomes don't matter for which scenario you're in. Two very different paths — e.g. "stealth Iranian campaign, no US response, mild damage" vs "major US operation, mild damage" — collapse to the same scenario distribution if they induce similar $\pi$.

Whether that's right is an architectural call. It's defensible if you treat $(D, T, P)$ as sufficient statistics for the scenario classification. It's wrong if you think kinetic causation matters independently of outcomes.

#### Is it actually a problem?

The math is internally consistent. The objection is about **framing and interpretability**, not correctness:

1. The scenario number is a *summary statistic*, not an independent inferential quantity. Stakeholders reading it as "the model's headline prediction" are reading it slightly wrong.
2. The CI machinery spends compute resampling a CPT ($f$) whose perturbation contributes only marginally to the final uncertainty.
3. The labeling function $f$ is a separable design artifact that deserves its own audit surface — currently it's mixed into `src/network.py` alongside the genuinely causal CPTs.

#### Three responses, increasing in surgery

1. **Document and live with it.** Add a paragraph to the Appendix making the classifier framing explicit. Lowest cost. Useful regardless of any other choice.
2. **Surface the joint $\pi(d, t, p \mid E)$ in the UI.** Add a "scenario decomposition" panel showing which parent configurations carry the mass. Lets stakeholders see *why* a scenario is at 42%. Compounds well with A3 (sensitivity attribution).
3. **Move $S$ out of the BN.** Make scenario probabilities a post-processing readout. Removes one CPT from the elicitation surface. Lets the labeling function be edited and version-controlled separately (e.g. "Scenario v2: tighter Stress_Mitigates definition"). Largest refactor.

Recommendation: (1) immediately and (2) as part of A3 work. (3) is worth doing if you find yourself wanting to A/B-test scenario definitions.

---

### M2. Soft-evidence semantics: translator output treated as likelihood, not posterior belief

**Severity.** 🔴 P0
**Files.** `src/inference.py:123-139`, `src/translator.py:106-122`
**Status.** Open

**What's wrong.** `_virtual_evidence_cpds()` packs the translator's `{state: prob}` dict into a single-column TabularCPD. pgmpy's virtual-evidence convention multiplies these weights into the marginal — i.e. they are interpreted as `P(obs | node = s)`, the likelihood of having observed *this evidence* given each state.

But the translator is prompted to produce a "probability distribution over states for this node. Must sum to 1.0" — the LLM is producing what it believes the node's distribution should be given the headline, i.e. a posterior-shaped quantity. Treating it as likelihood means:

```
posterior(s) ∝ prior(s) × translator_dist(s)
```

The prior compounds with the LLM's belief. A translator output of `{none: 0.05, isolated: 0.15, frequent: 0.80}` with prior `P(Tanker=frequent) ≈ 0.10` produces a posterior around **~60–70%** on `frequent`, not 80%.

**Why it matters.** For symmetric inputs the difference is small; for asymmetric ones (rare states with strong evidence, or common states with weak evidence) the divergence is material. Every CI in the UI is in a slightly wrong place. This is the single most consequential semantic mismatch in the codebase.

**Suggested direction.** Pick one and document it clearly:

1. *Reprompt for likelihoods.* Ask the translator for `P(headline | node = s)` per state. This is what virtual evidence actually assumes.
2. *Pre-divide by prior client-side.* Convert posterior-shaped output to implied likelihood via `likelihood(s) = translator(s) / prior(s)` before injecting.
3. *Keep current behaviour, document explicitly.* If the compounding-with-prior effect is desired (priors are sticky, translator only nudges), say so in the Appendix.

---

### M3. Dirichlet concentration κ=20 is uniform across every CPT

**Severity.** 🟡 P1
**Files.** `src/sensitivity.py:53`
**Status.** Open

**What's wrong.** A single hyperparameter (`concentration=20.0`) drives every credible interval in the UI. Problems:

- No motivation in code or docs. Why 20 vs 10 vs 100? Dirichlet variance scales roughly as `θ(1-θ)/(κ+1)`, so κ=20 puts the implied standard deviation on a 50/50 entry at ~10pp. That's a strong claim about expert calibration.
- Some CPTs deserve much smaller κ. The `Scenario | damage, duration, path` CPT is purely definitional — no empirical grounding. Treating it with the same κ as `Tanker_Incidents | militia, negotiations` (which has plausible historical analogues) is wrong direction.

**Why it matters.** The 🟢🟡🔴 robustness badges and all reported CIs flow from this. A choice that no one elicited drives stakeholder-facing claims.

**Suggested direction.** Two complementary changes:

1. Allow per-CPT κ. Store it next to each CPD or derive it from a coarse "evidence basis" tag (`empirical`, `expert_elicited`, `definitional`).
2. Expose a global κ slider in the Appendix/expert mode, so a user can see how robust scenario claims are under varying elicitation tightness.

---

### M4. Within-CPT columns sampled independently — no correlated shape uncertainty

**Severity.** 🟡 P1
**Files.** `src/sensitivity.py:26-28`
**Status.** Open

**What's wrong.** Each column of each CPT gets its own `rng.dirichlet` draw. This implicitly assumes the analyst could have gotten any individual column wrong independently of the others. In reality, if you systematically underweight "militia high → tanker frequent," you very likely also underweight "militia elevated → tanker frequent" — the columns are correlated by the underlying mechanism.

**Why it matters.** The current method captures **noise** uncertainty but not **shape/structural** uncertainty. CIs may look narrower than they should for the kind of misspecification a stakeholder would actually worry about ("what if our whole reading of militia tempo is biased?").

**Suggested direction.** Introduce a per-CPT row-scaling perturbation: sample one noise vector per CPT and apply it consistently across all columns of that CPT, then renormalize. Or expose two sensitivity modes side by side: independent (noise) and correlated (shape).

---

### M5. DAG omissions called out in UI are real channels, not pedantic

**Severity.** 🟡 P1
**Files.** `app/dashboard.py:1575-1604` (the `_EDGE_OMISSIONS` list)
**Status.** Open

**What's wrong.** The most consequential admitted omission is:

> `Iran_Aligned_Militia_Attacks → US_Military_Response`: "Known limitation — US base attacks in Iraq/Syria have historically triggered direct US strikes without any tanker incident."

That's not a minor mediation gap — it's the dominant historical pattern for kinetic US responses (Soleimani strike, Tower 22 retaliation). The current DAG forces all militia-driven escalation through tanker incidents.

Also questionable:

- No edge `Oil_Price_Regime → Scenario`. A sustained $150 oil environment is part of what Severe_Closure *means* operationally; routing it only through Damage + Duration + Path drops a salient signal.
- No edge `Energy_Infrastructure_Damage → Conflict_Duration`. Damage to terminals raises the political cost of de-escalation; the current model has duration depending only on negotiations + mediation + military response.

**Why it matters.** The model is currently biased toward a Hormuz-flavored maritime escalation. Stakeholders looking at non-maritime escalation triggers (Iraq/Syria base strikes, broader regional kinetic events) will see no model response until those events somehow propagate through Tanker_Incidents.

**Suggested direction.** Either:

1. Add the missing edges with conservative-weight CPTs (and update `_EDGE_RATIONALE`).
2. Scope the model explicitly to "maritime-channel escalation" in the README, the Appendix, and the dashboard header. Either is fine; what's not fine is the current ambiguity.

---

### M6. Root priors are unjustified and propagate everywhere

**Severity.** 🟡 P1
**Files.** `src/network.py:100-120`
**Status.** Open

**What's wrong.** `P(Sanctions=tightening)=0.30`, `P(Negotiations=stalled)=0.55`, etc. These are the *base rates* for the whole model — the prior `Severe_Closure` headline number is fully determined by them combined with the topology. There's no:

- Citation, calibration date, or expert-name attribution for each prior.
- Sensitivity report on what happens if any of these moves ±10pp.
- Explicit decision on whether priors should be re-elicited periodically.

**Why it matters.** For a stakeholder-facing tool, the prior is the most arguable part of the model. The first question in any defense-committee meeting will be "where do these numbers come from?" — and the current answer is "a comment in the source file."

**Suggested direction.**

1. Add a `priors_provenance` block (YAML or markdown) listing each prior, the date elicited, the elicitor, and a one-line rationale.
2. Add a "prior sensitivity" panel to the Appendix showing how scenario probabilities shift if any single prior moves ±10pp. Cheap to compute (5 inferences per prior × 4 priors).
3. Add a calendar reminder convention (90-day or 180-day) to re-elicit.

---

### M7. Resample-mean vs point-estimate posterior used inconsistently in UI

**Severity.** 🟡 P1
**Files.** `app/dashboard.py:1099-1104`, `app/dashboard.py:1402-1417`, `app/dashboard.py:1838`
**Status.** Open

**What's wrong.** The dashboard deliberately uses the Dirichlet-resample mean as the displayed scenario probability so the band is centered. Defensible — but:

- The **Posterior panel** for non-observed nodes uses the resampled mean (reads from `node_ci_table`), while observed nodes use point-estimate values from `engine.get_node_marginal()` (via `_flat_bar_chart`).
- The **probability-evolution chart** uses resampled mean (good — matches the band).
- The **intermediate-node marginals table in the Audit tab** uses `engine.get_node_marginal(node)` — the point estimate.

Two different numbers can appear in different tabs for the same node, off by 1–3pp. See also V1, which is the user-facing manifestation.

**Why it matters.** Inconsistency between tabs erodes stakeholder trust. The first time someone notices "the network tab says 42% but the audit tab says 44%," the whole tool reads as buggy.

**Suggested direction.** Pick one and commit:

- Either everywhere use resample-mean (consolidate into one function that returns mean + CI per node).
- Or everywhere use point-estimate (and treat the resample only as bands/CIs, never as a central tendency).

---

### M8. `+1e-6` Dirichlet alpha guard never fires meaningfully

**Severity.** 🟢 P2
**Files.** `src/sensitivity.py:27`
**Status.** Open

**What's wrong.** `alpha = concentration * values[:, col] + 1e-6` is documented as "avoid zero-alpha." For any elicited 0.01, this is effectively no change. The current CPTs have a minimum value of 0.01, so the guard never fires.

**Why it matters.** Low-stakes — but it's an unexplained constant that could mask a real bug if a future CPT edit puts a true 0 into the table.

**Suggested direction.** Either remove for clarity, or document explicitly as a defensive measure against future CPT edits introducing structural zeros.

---

### M9. `scenario_credible_intervals` and `node_credible_intervals` duplicate the resampling loop

**Severity.** 🟢 P2
**Files.** `src/sensitivity.py:48-84`, `src/sensitivity.py:104-181`
**Status.** Open

**What's wrong.** The two functions implement essentially the same Monte-Carlo procedure twice; `test_node_ci_matches_scenario_ci_for_scenario_node` verifies they agree numerically. The scenario-only function is now redundant — `node_credible_intervals(..., nodes=["Scenario"])` does the same job.

**Why it matters.** Two implementations means two surfaces to keep in sync. Mostly maintenance overhead.

**Suggested direction.** Have `scenario_credible_intervals` forward to `node_credible_intervals(..., nodes=["Scenario"])`, or delete it entirely and update callers. The test stays as a regression guard if you keep the wrapper.

---

## Code Implementation

### C1. `engine` is `@st.cache_resource` but is mutated on every rerun

**Severity.** 🔴 P0
**Files.** `app/dashboard.py:451-453`, `app/dashboard.py:895-901`, `app/dashboard.py:1043`
**Status.** Open

**What's wrong.**

```python
@st.cache_resource
def get_engine() -> BNInferenceEngine: ...

engine = get_engine()
engine.clear_evidence()
engine.update_evidence(evidence)
```

`cache_resource` returns the same object across all sessions and users in a Streamlit deployment. Mutating it (`clear_evidence`, `update_evidence`) creates cross-session bleed and an interleaving race if two users interact concurrently.

The same pattern repeats inside the probability-evolution loop with `engine_h = get_engine()` — same singleton, same mutation.

**Why it matters.** Works fine on your laptop with one tab open. Breaks in any deployment beyond that — two stakeholders demoing concurrently could see each other's evidence.

**Suggested direction.** Two acceptable patterns:

1. Make queries pure functions of evidence — `query(net, evidence)` rather than `engine.update_evidence(...); engine.query()`. Cache only the network build.
2. Move the engine into `st.session_state` (per-session) and keep `cache_resource` only on the underlying `DiscreteBayesianNetwork`.

---

### C2. `dashboard.py` is 1878 lines / 76 KB in a single file

**Severity.** 🔴 P0
**Files.** `app/dashboard.py` (entire)
**Status.** Open

**What's wrong.** CSS (~350 lines), session-state plumbing, four tabs of UI, helper components (CI dataframes, badge HTML, dumbbell charts, robustness logic), and computation are all interleaved.

**Why it matters.** Prereq for both maintainability and for unit-testing the helpers. Anything that touches one part risks regressing another. Onboarding cost: high.

**Suggested direction.** Concrete decomposition:

- `app/styles.css` — load via `st.markdown(open(...).read(), unsafe_allow_html=True)`.
- `app/components/` — `scenario_cards.py`, `ci_charts.py`, `network_tab.py`, `observation_log.py`, `audit_tab.py`, `edge_rationale_tab.py`.
- `app/state.py` — session-state defaults, named-session save/load, `_merged_evidence`, `_append_observation`.
- `app/dashboard.py` — orchestration only.

Pure refactor, no behavior change. Lets you add tests for `_width_category`, `_robustness_badge_html`, `_merged_evidence`, etc.

---

### C3. Probability-evolution chart recomputes full posterior series on every rerun

**Severity.** 🟡 P1
**Files.** `app/dashboard.py:1042-1090`
**Status.** Open

**What's wrong.** For each day, the block rebuilds cumulative evidence, runs inference, and (via `cached_credible_intervals`) does m=200 Dirichlet draws. A 10-day session triggers 10 day-level cache lookups every rerun — fine on cache hits, but the cache key includes evidence values, so any slider tick on the override panel busts the relevant key.

**Why it matters.** Laggy interactions for long sessions. Once you have 15+ observations and are tweaking sliders in the override panel, every change triggers visible recomputation.

**Suggested direction.** Two fixes:

1. Memoize the long-form `long_df` itself, keyed on the (sorted) observation IDs. Each new observation invalidates only the trailing days.
2. Or maintain the long-form history as session state, appending one row per new observation rather than recomputing the entire series.

---

### C4. Streamlit caches have no bounds

**Severity.** 🟡 P1
**Files.** `app/dashboard.py:456-474`
**Status.** Open

**What's wrong.** Both `cached_credible_intervals` and `cached_node_credible_intervals` are `@st.cache_data` without `max_entries` or `ttl`. Over a long demo session with frequent slider tweaks, the cache grows unboundedly.

**Why it matters.** Memory leak in long-running deployments. Each unique evidence configuration takes ~tens of KB.

**Suggested direction.** Add `max_entries=64, ttl="1h"` to both decorators.

---

### C5. Soft evidence as virtual-evidence-CPD encodes a likelihood mismatch

**Severity.** 🟡 P1
**Files.** `src/inference.py:123-139`
**Status.** Open

**What's wrong.** Mirror of M2 from the code side. The implementation is consistent with pgmpy's semantics; the mismatch is at the *interface* between translator (produces posterior-shaped distributions) and inference (expects likelihood).

**Why it matters.** Single-line fix once the decision in M2 is made. Tracked separately so you can close one without the other.

**Suggested direction.** Once M2 is resolved, this becomes either (a) leave as-is (current code matches the documented decision), or (b) divide by prior before constructing the virtual-evidence CPD.

---

### C6. Translator validation accepts dict / list / string forms of `state_probs`

**Severity.** 🟡 P1
**Files.** `src/translator.py:235-289`
**Status.** Open

**What's wrong.** Three parse paths for the same field — anti-pattern. The OpenAI strict-schema constrains it to an array of `{state, prob}` objects, but the Claude Code path uses an unconstrained system prompt so the model can return any of the three shapes. The "accept everything and normalize" path silently coerces malformed LLM output.

**Why it matters.** Hardest path to debug: the validator says "validated 1 assignment" but you have no idea what the model actually returned. Prompt drift is invisible.

**Suggested direction.** Either:

- Constrain Claude Code via a tool-use schema (claude-agent-sdk supports this), or
- Document which form is canonical in the prompt and reject the others with a clear error.

---

### C7. `_validate_payload` re-normalizes any positive-sum probability vector silently

**Severity.** 🟡 P1
**Files.** `src/translator.py:288-289`
**Status.** Open

**What's wrong.**

```python
total = sum(probs.values())
probs = {k: v / total for k, v in probs.items()}
```

An LLM that returns `[0.99, 0.99, 0.99]` silently becomes `[0.33, 0.33, 0.33]`. The user is told "validated 1 assignment" but the translator has effectively produced no signal.

**Why it matters.** Quality-of-evidence degradation goes unflagged. A consistently bad LLM looks the same as a good one in the audit trail.

**Suggested direction.** Reject sums outside `[0.95, 1.05]` with a clear error that surfaces in the translator-error display.

---

### C8. `_extract_json_block` regex is fragile

**Severity.** 🟡 P1
**Files.** `src/translator.py:297-312`
**Status.** Open

**What's wrong.** `re.search(r"\{.*\}", text, flags=re.DOTALL)` is greedy and breaks if the LLM emits a code-fenced block plus surrounding prose, or two JSON objects (e.g. `{"reasoning": ...} {"assignments": ...}`).

**Why it matters.** Currently masked by the strict-mode OpenAI path. The Claude Code path goes through this regex and is exposed.

**Suggested direction.** Use `json5` or a brace-matching parser. Or constrain Claude Code via tool-use schema (resolves both C6 and C8).

---

### C9. `render_network_png` is dead code

**Severity.** 🟡 P1
**Files.** `src/viz.py:156-211`
**Status.** Open

**What's wrong.** Built for Streamlit display, but the dashboard's network tab uses `build_agraph_payload` exclusively. The PNG renderer would be useful for B3 (session export) — keep it only if you commit to wiring it up.

**Why it matters.** ~60 lines of HTML-label code that no one is testing or maintaining.

**Suggested direction.** Either delete now, or fold into the B3 export work when that lands. Add a `# pragma: no cover` note if keeping interim.

---

### C10. `_NODE_LEVEL` in `viz.py` is a manually maintained DAG layout

**Severity.** 🟡 P1
**Files.** `src/viz.py:229-243`
**Status.** Open

**What's wrong.** Hardcoded layout levels for the hierarchical agraph view. If a node is added to `STATES` but not added here, the agraph layout silently defaults to level 0.

**Why it matters.** Coupling between network topology and visualization. Any future schema change requires manually editing this dict.

**Suggested direction.** Compute levels from `EDGES` at import time (longest path from any root). Memoize.

---

### C11. No test for `_merged_evidence` soft↔hard ordering invariants

**Severity.** 🟢 P2
**Files.** `app/dashboard.py:571-582`, `tests/`
**Status.** Open

**What's wrong.** The logic correctly pops a node from the other dict when a new observation switches its evidence type — but there's no test. Easy to break in a refactor.

**Why it matters.** A regression here would silently double-count evidence on a node.

**Suggested direction.** After C2 (split), extract `_merged_evidence` to `app/state.py` and add an ordering-invariants test:

- hard → soft → hard: final state hard.
- soft → hard → soft: final state soft.
- Two soft on different nodes: both preserved.

---

### C12. `_PLUGINS_REGISTERED` workaround needs a TODO

**Severity.** 🟢 P2
**Files.** `src/viz.py:22-40`
**Status.** Open

**What's wrong.** Subprocess call to `dot -c` is a workaround for a conda-forge graphviz packaging bug. The comment explains it well, but there's no signal for when to remove it.

**Why it matters.** Workaround code accretes if no one ever revisits it.

**Suggested direction.** Add `# TODO: remove once conda-forge graphviz ships a populated plugin config (tracked: [link])`.

---

### C13. `Observation.tone` is declared but unused

**Severity.** 🟢 P2
**Files.** `src/evidence.py:15`, `src/evidence.py:33-36`
**Status.** Open

**What's wrong.** `Tone` is a typed `Literal` used only by `ExampleHeadline`. The `Observation` dataclass doesn't carry tone.

**Why it matters.** Minor noise in the type vocabulary.

**Suggested direction.** Either thread tone through observations (could feed into B1 narrative generation) or drop the type.

---

### C14. No tests for `viz.py` or dashboard helpers

**Severity.** 🟢 P2
**Files.** `tests/`
**Status.** Open

**What's wrong.** Good coverage of network/inference/translator. Zero coverage of:

- `_merged_evidence` ordering (see C11).
- `_width_category` / `_robustness_badge_html` thresholds.
- Save/load session round-trip.
- `build_agraph_payload` (at least: returns the right node count, every observed node has the observed-fill color).

**Why it matters.** The visualization and state logic are the parts most likely to silently regress on a refactor.

**Suggested direction.** Cheap once C2 is done. Add a `tests/test_dashboard_helpers.py` after the split.

---

## Visualization

### V1. Headline scenario card numbers are resample-mean, not point-estimate

**Severity.** 🔴 P0
**Files.** `app/dashboard.py:952-963`
**Status.** Open

**What's wrong.** Stakeholders read the giant "42.3%" and reasonably infer "this is the model's answer." It is actually `E_resample[P(Scenario=s | E)]`, which can differ from the point-estimate `P(Scenario=s | E)` because inference is non-linear in the CPTs.

**Why it matters.** The choice is deliberate (so the value sits centered in the CI band) but invisible to the reader. Currently the same node can show one number on the card and a different number in the audit table without explanation. See also M7 for the internal inconsistency.

**Suggested direction.** Either:

- Add a small "ⓘ" affordance on the card explaining "mean across CPT-resample samples; the point-estimate posterior is X%."
- Or switch to point-estimate everywhere and put the resample-mean only inside the CI panel.

---

### V2. Information density is wrong: three giant cards for three numbers

**Severity.** 🟡 P1
**Files.** `app/dashboard.py:946-1026`
**Status.** Open

**What's wrong.** The scenario cards eat ~1/3 of the viewport. Each contributes a single percentage + a one-line CI + a paragraph of narrative that doesn't change with evidence. The evolution chart — the only thing that actually moves with new evidence — is comparatively small.

**Why it matters.** The chart is the story. The cards are wallpaper after two demos.

**Suggested direction.** Compress the cards to a vertical strip (1/5 viewport) and give the saved space to the evolution chart.

---

### V3. Robustness thresholds (±8, ±20 pp) are arbitrary; sharp emoji transitions

**Severity.** 🟡 P1
**Files.** `app/dashboard.py:1183-1188`
**Status.** Open

**What's wrong.** A node with widest CI 7.9 pp is 🟢 robust; 8.1 pp is 🟡 moderate. The badge is prominent in the UI but a small numerical change flips the color.

**Why it matters.** Stakeholders will fixate on color transitions and read meaning into them that the thresholds don't support.

**Suggested direction.** Either:

- Smooth color interpolation (LERP between green/amber/red along the half-width) instead of three buckets.
- Or calibrate thresholds against the actual half-width distribution across all nodes at the prior — define "fragile" as the worst quartile *in this network*, not an external constant.

---

### V4. Manual override sliders are a UX trap

**Severity.** 🟡 P1
**Files.** `app/dashboard.py:1431-1486`
**Status.** Open

**What's wrong.** Three independent 0-100 integer sliders, primary button disabled until they sum to exactly 100. The user has to do mental arithmetic.

**Why it matters.** The override panel is the main "what-if" exploration tool. Friction here directly reduces exploratory use, which is the most valuable mode for committee demos.

**Suggested direction.** Two better patterns:

- *Drag-to-simplex*: move slider A, the rest auto-rescale proportionally so total always = 100.
- *Anchor + distribute*: let the user enter one state value (e.g. `frequent: 70%`) and auto-fill the others proportionally to the current prior.

---

### V5. Hard-observed nodes show an uninformative bar chart

**Severity.** 🟡 P1
**Files.** `app/dashboard.py:1252-1280`, `app/dashboard.py:1402-1406`
**Status.** Open

**What's wrong.** `_flat_bar_chart` shows 100% on one state, 0% on the others, in plain navy. The bar is redundant when one state is 1.0.

**Why it matters.** The detail panel for the most interesting nodes (the ones you just set evidence on) is the least informative.

**Suggested direction.** Replace with a richer card:

- "Observed: `frequent` (Day 5, via translator, headline: '...')"
- "Downstream effect: shifted Severe_Closure by +18 pp" (this is essentially A3 attribution).
- Possibly: "Posterior on this node *before* the observation: ..."

---

### V6. DAG layout: level 3 has four stacked nodes, level 4 has only Scenario

**Severity.** 🟡 P1
**Files.** `src/viz.py:229-243`
**Status.** Open

**What's wrong.** The right side of the graph crowds vertically at one x-position while Scenario sits alone at the next level. `Oil_Price_Regime` has no outgoing edge to Scenario — it's actually a terminal sibling and could move to its own level.

**Why it matters.** Visual clutter on the most-watched part of the graph.

**Suggested direction.** Move `Oil_Price_Regime` to level 4 (parallel to Scenario, not before it). Or, once C10 is done, compute levels topologically.

---

### V7. Evolution chart conflates parameter uncertainty with forecast uncertainty

**Severity.** 🟡 P1
**Files.** `app/dashboard.py:1131-1134`, `app/dashboard.py:1154-1161`
**Status.** Open

**What's wrong.** The shaded band is "80% CI on the posterior *at that day's evidence state*" — a measure of model robustness. But it looks identical to a forecast confidence band, which would represent "what might the probability be next week." The caption calls this out, but the visual is misleading regardless.

**Why it matters.** Anyone glancing at the chart without reading the caption will misread the band.

**Suggested direction.**

- Render the band differently (hashed fill, or dotted edges) to flag "parameter uncertainty, not forecast uncertainty."
- Add a small fixed-y-position annotation: "Band = robustness; not a forecast."

---

### V8. ~350 lines of CSS embedded inline

**Severity.** 🟡 P1
**Files.** `app/dashboard.py:88-443`
**Status.** Open

**What's wrong.** Embedded `<style>` block dominates the top of the dashboard module.

**Why it matters.** Maintainability. Edits to style require scrolling past 350 lines of `!important` rules. Lints and formatters can't help.

**Suggested direction.** Move to `app/static/app.css` and load via `st.markdown(open(...).read(), unsafe_allow_html=True)`. Done as part of C2.

---

### V9. No before/after visualization on a new observation

**Severity.** 🟢 P2
**Files.** `app/dashboard.py` (no current implementation)
**Status.** Open

**What's wrong.** When a headline is committed the probabilities shift, but there's no diff: no "+8pp Severe_Closure," no animation, no callout. The user has to read both numbers and subtract.

**Why it matters.** This is exactly what A3 in the roadmap addresses. Listed here for completeness — until A3 lands, even a "since last update: ▲ +5pp Severe" chip on each scenario card would close most of the gap.

**Suggested direction.** Minimal version: store the previous scenario probabilities in session state, render a small delta chip on each card. Full version: ship A3.

---

### V10. Latest-translation panel renders distributions as plain text

**Severity.** 🟢 P2
**Files.** `app/dashboard.py:1687-1700`
**Status.** Open

**What's wrong.** A 3-state distribution as `"none: 5% · isolated: 15% · frequent: 80%"` is harder to scan than a 60-pixel-wide horizontal stacked bar.

**Why it matters.** The Observations tab is where stakeholders inspect "did the translator read this headline correctly?" Easier to read = more scrutiny = more trust.

**Suggested direction.** Render an inline stacked bar per assignment; keep the text as a tooltip.

---

### V11. Green/red distinction is colorblind-hostile

**Severity.** 🟢 P2
**Files.** `app/dashboard.py:67-75`
**Status.** Open

**What's wrong.** `GREEN #2E8B57` (Stress_Mitigates) vs `RED #B22222` (Severe_Closure) collapses under deuteranopia. The evolution chart's three lines are distinguishable only by hue.

**Why it matters.** Roughly 5% of male stakeholders won't be able to read the chart. For a committee tool, that's not zero.

**Suggested direction.**

- Add shape/style encoding on lines and points (solid / dashed / dotted).
- Or switch to a CVD-safe palette: e.g. `#0072B2`, `#E69F00`, `#D55E00`.

---

### V12. Headline truncation on evolution-chart tooltip is aggressive (180 chars)

**Severity.** 🟢 P2
**Files.** `app/dashboard.py:1074-1075`
**Status.** Open

**What's wrong.** 180 chars truncate to "...". For 3+ headlines on a day this clips most of the second one.

**Why it matters.** Tooltip is the main "what happened that day?" affordance on the chart.

**Suggested direction.** Drop the truncation; Altair tooltips handle wrapping. Or list each headline on its own line.

---

### V13. Edge-rationale tab is isolated from the graph

**Severity.** 🟢 P2
**Files.** `app/dashboard.py:1611-1644`
**Status.** Open

**What's wrong.** The edge rationale lives in a separate tab; the DAG can't surface it on hover.

**Why it matters.** "Why this edge?" is a natural question while looking at the graph, but answering it requires switching tabs.

**Suggested direction.** Two options:

- Click/hover an edge in the network → tooltip shows the rationale (`streamlit_agraph` supports edge titles).
- Add per-edge "ⓘ" affordance in the Audit tab.

---

### V14. Scenario cards have a fixed narrative paragraph that doesn't change with evidence

**Severity.** 🟢 P2
**Files.** `app/dashboard.py:955-961`
**Status.** Open

**What's wrong.** The narrative is the same on Day 0 as on Day 30. Decoration after two demos.

**Why it matters.** Wastes prime visual real estate (see V2).

**Suggested direction.** Either:

- Remove from the always-on card; put on hover/expander.
- Use B1 (daily narrative generation) to make the narrative responsive.

---

### V15. 460px DAG canvas height is calibrated to the right column — fragile coupling

**Severity.** 🟢 P2
**Files.** `src/viz.py:411-412`
**Status.** Open

**What's wrong.** `height=460` "calibrated to the combined height of the Posterior and Override boxes." If the override sliders gain a tooltip or the badge wraps to two lines, the DAG either underflows or overflows.

**Why it matters.** Brittle layout that breaks on small UI changes elsewhere.

**Suggested direction.** Flex/grid layout, or measure the right column at runtime. Lower priority — works fine in practice today.

---

## Recommended Sequencing

If compressed into a sequencing for this branch, the suggested order is:

1. **M2 / C5 — soft-evidence semantics.** Decide and document. One paragraph in `model_documentation.md` plus possibly one code change. Without this, every CI in the UI is in a slightly wrong place.
2. **C2 — split `dashboard.py`.** Prereq for both maintainability and for unit-testing the helpers. Pure refactor, no behavior change.
3. **C1 — engine cache fix.** One-line move from `cache_resource` to `session_state`. Prevents cross-user state bleed.
4. **M7 / V1 — resample-mean vs point-estimate reconciliation.** Pick one consistently and label it.
5. **M3 — expose κ as a UI control.** Lets stakeholders see how much "robustness" depends on a hyperparameter no one elicited.
6. **M1 — document the Scenario-as-classifier framing.** Lowest-cost option: a paragraph in the Appendix. Higher-cost options can come later.

After these foundational items, proceed with the roadmap as planned. A1 (evidence accumulation) is the right next foundational *feature*, exactly as `bn_app_next_steps.md` says — but the items above are at a different level: they're about hidden semantic choices in the existing code that should be either fixed or documented as deliberate before shipping more on top.

The remaining P1 and P2 items can be tackled opportunistically (most are <1 hour each), or batched alongside related feature work (e.g. V9 lands naturally with A3; V8 with C2; C14 with C2).
