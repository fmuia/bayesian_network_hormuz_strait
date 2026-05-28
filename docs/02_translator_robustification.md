# Translator Robustification: Plan to Move from Demo to Tool

## Executive Summary

**What the translator is.** The translator (`src/translator.py`) is the bridge between the analyst and the Bayesian network. The analyst pastes a news headline (*"Iran fires missile at US carrier"*); the translator decides which network variables the headline speaks to (e.g., militia activity, military response), assigns a distribution over each variable's possible states, returns a short rationale, and hands the result to the BN engine as evidence. It runs once per headline and is the only place natural language enters the model.

**Why today's translator is a demo, not a tool.** It works well enough on stage, but eight concrete problems make it unsafe for stakeholder-facing decisions:

1. **Wrong math at the interface (M2/C5).** The prompt asks the LLM for a probability distribution summing to 1 — but the BN engine consumes the output as a likelihood, not a posterior. The two are different mathematical objects, and confusing them double-counts the network's prior. Every node marginal and scenario percentage computed *under translator-injected evidence* — i.e., every number on the dashboard once the analyst has added at least one observation — is biased as a result. Prior marginals (no observations applied) are unaffected.
2. **Only the headline goes in.** Source identity, body, qualifiers (*"unconfirmed," "no injuries"*) are stripped before the LLM sees anything. The accuracy ceiling is whatever a single sentence can carry.
3. **One shot, zero measurement of own confidence.** A single LLM call at zero temperature produces one answer. There is no measurement of how confident the model itself is in its output — the displayed confidence numbers are hand-rolled by the LLM with no evidentiary basis.
4. **No "I don't know" path.** Off-topic headlines still produce confident-looking assignments; there is no enforced relevance check.
5. **Validator hides drift (C6/C7/C8).** The validator accepts three different input shapes, silently renormalises any positive sum (so `[0.99, 0.99, 0.99]` quietly becomes uniform), and extracts JSON with a greedy regex. Bad LLM output is indistinguishable from good output in the audit trail.
6. **Prompt is invisible.** The system prompt is inlined Python regenerated at runtime, with no version, owner, or changelog. Edits ship with no test gate.
7. **No measurement.** There is no labelled test set, no per-node accuracy number, no calibration data. The question *"does the translator work?"* has no defensible answer.
8. **No audit trail.** Article URL, source credibility, prompt version, raw-response hash, analyst-approval state — none of it is persisted. Reproducibility is best-effort.

Each problem is concrete and cited line-by-line in the Diagnosis section below.

**What the plan delivers.** Twelve items across five themes, sequenced as 13 execution slots (item B1 splits into B1a/B1b):

- **A — Foundations.** Pick a single mathematical semantics for the translator-to-BN interface (likelihood ratios; fixes problem 1) and harden the input/output schema (fixes 5).
- **B — Reasoning.** Give the translator the full article rather than just the headline (fixes 2); make it cite verbatim source text for every assignment so hallucination becomes structurally impossible; add an enforced *"not relevant"* path (fixes 4).
- **C — Uncertainty.** Replace the LLM's hand-rolled confidence numbers with measured confidence — multi-sample self-consistency (the same article translated several times, disagreement becomes the uncertainty signal) and optional multi-model cross-check (fixes 3).
- **D — Governance.** Externalise the prompt as a versioned file behind a CI test gate (fixes 6); build a 30→500-record labelled evaluation set with per-node accuracy and calibration plots (fixes 7); persist a full audit log keyed by response hash (fixes 8).
- **E — Operations.** Borderline translations route to a human-in-the-loop review queue rather than auto-committing; the audit log doubles as a retrieval corpus that injects past analyst-approved translations as in-context examples for new articles.

**Sequencing logic.** The first three slots — A1 (semantics fix), A2 (schema hardening), D2-MVP (30-record labelled set) — are a **minimum viable correctness baseline**: until they ship, nothing later in the plan is measurable. Slots 4–7 (article-level input, relevance filter, span-grounded structured reasoning, self-consistency ensemble) deliver the largest accuracy and calibration jump — article-level input is slot 4 specifically so that B2's span-grounding and B3's relevance pre-filter both work against full article text rather than the headline. Slots 8–13 build the institutional layer — versioning, audit, analyst review, retrieval memory — that distinguishes a tool from a script.

**Status legend.** ✅ = shipped. Nothing yet.

## Context

The translator sits between an analyst typing a headline into the dashboard and the BN engine running variable elimination. It has three responsibilities:

1. Decide which BN nodes the article speaks to.
2. For each such node, produce a soft distribution over that node's states.
3. Return a rationale a human can audit.

Output is consumed by [src/inference.py:123-139](src/inference.py#L123-L139), which wraps it in pgmpy `TabularCPD`s and uses pgmpy's virtual-evidence convention. The translator is provider-pluggable (Claude Code / OpenAI; see [src/translator.py:198-205](src/translator.py#L198-L205)) and is invoked once per headline from the dashboard.

The current implementation is ~509 lines: schema definition, system-prompt construction, two provider backends, a permissive validator, and a dispatcher. It has unit coverage of the validator but no coverage of translation quality.

This plan operates one layer below the existing roadmap in [docs/bn_app_next_steps.md](docs/bn_app_next_steps.md): the roadmap addresses how evidence *accumulates* and is *narrated*; this plan addresses how evidence is *produced*. The two documents share a downstream item — the news memory database E1 — which appears in both as a reach goal.

---

## Diagnosis: Why the Current Translator Is a Demo

The list below is the failure surface this plan closes. Items marked (M*/C*) are finding IDs from the master-plan §4 matrix and are included for completeness.

1. **Semantic mismatch with the inference layer (M2, C5).** The prompt at [src/translator.py:106-122](src/translator.py#L106-L122) asks for a *posterior-shaped* distribution that sums to 1. The inference layer at [src/inference.py:123-139](src/inference.py#L123-L139) consumes it as a *likelihood*. Every node marginal and credible interval computed under translator-injected evidence is in a slightly wrong place because of this single interface bug; prior marginals (no observations applied) are unaffected.
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

**What exists now.** The system prompt at [src/translator.py:106-122](src/translator.py#L106-L122) instructs the LLM to produce a "probability distribution over states for this node" that "must sum to 1.0." This is a **posterior-shaped** output ($P(s \mid A)$ where $s$ is a state and $A$ is the article). The inference layer at [src/inference.py:123-139](src/inference.py#L123-L139) wraps it in a single-column `TabularCPD` and passes it via pgmpy's `virtual_evidence` parameter, which interprets the values as **likelihoods** ($P(V=v \mid N=s)$ for a virtual child $V$). The two mathematical objects are not the same, and the mismatch causes pgmpy to double-count the BN's prior on every translated headline.

**The math: why the current interface is wrong.**

*The two quantities at play.* When a node $N$ has states $s_1, \ldots, s_k$ and you have evidence $A$ (the article), two distinct objects are in scope:

- **Likelihood** $P(A \mid s_i)$ — *"how plausible is this article if the true state were $s_i$?"* A function of $s_i$ for fixed $A$. Does **not** sum to 1 over $i$; it's a relative quantity in $[0, \infty)$.
- **Posterior** $P(s_i \mid A)$ — *"what's my belief about which state is in force, having seen the article?"* **Does** sum to 1 over $i$.

Bayes' rule connects them:
$$P(s_i \mid A) \;=\; \frac{P(A \mid s_i) \cdot P(s_i)}{P(A)}$$

i.e., **posterior = likelihood × prior / normaliser**.

*What the translator produces today.* The prompt instructs *"state_probs must include ALL allowed states for that node and probabilities must sum to 1.0"* ([src/translator.py:158-161](src/translator.py#L158)). The LLM, asked for a sum-to-1 distribution, is producing its best estimate of the posterior $T_i = P(s_i \mid A)$.

*What pgmpy's `virtual_evidence` expects.* pgmpy's `BeliefPropagation.query(virtual_evidence=...)` and `VariableElimination.query(virtual_evidence=...)` both implement **Pearl's virtual evidence** (Pearl 1988, *Probabilistic Reasoning in Intelligent Systems*, Ch. 2) — a bookkeeping trick for injecting soft evidence into a BN without inventing a special soft-evidence algorithm. The construction: bolt a **fictional leaf node** $V$ onto $N$ as a phantom child (no real-world referent — no "article happened-ness" you could go measure), declare it observed at some value $v$, and fill its CPT column $P(V = v \mid N = s_i)$ with the per-state likelihoods you want to inject. That column is, by definition, a likelihood — a function of $s_i$ taking values in $[0, \infty)$, not summing to 1. Conditioning on $V = v$ then multiplies that likelihood into the joint via plain Bayes' rule:

$$P(N = s_i \mid V = v) \;\propto\; P(V = v \mid N = s_i) \cdot P(N = s_i)$$

pgmpy's `virtual_evidence` parameter is syntactic sugar over this construction: internally it adds the dummy child, marks it observed, runs the query, and drops the dummy from the result. You never see the phantom node, but the multiplication into the joint is real. **The CPT column you hand to pgmpy is interpreted as a likelihood, and pgmpy multiplies it by the prior to produce the posterior** — which is the contract A1 needs the translator to match.

*The bug: prior squaring.* The current code feeds pgmpy the translator's posterior $T_i = P(s_i \mid A)$. pgmpy treats it as a likelihood:
$$P(s_i \mid V = v) \;\propto\; T_i \cdot P(s_i) \;=\; P(s_i \mid A) \cdot P(s_i)$$

Expanding $T_i$ via Bayes' rule:
$$P(s_i \mid V = v) \;\propto\; \frac{P(A \mid s_i) \cdot P(s_i)}{P(A)} \cdot P(s_i) \;\propto\; P(A \mid s_i) \cdot P(s_i)^2$$

**The prior is squared.** We wanted $P(s_i \mid A) \propto P(A \mid s_i) \cdot P(s_i)$; we got $P(A \mid s_i) \cdot P(s_i)^2$. Every node-marginal in the BN that flows through a translated headline is computed with the prior counted twice.

*Caveat on the magnitude.* The derivation assumes the LLM faithfully emits $T_i = P(s_i \mid A)$ under *the BN's* prior $P(s_i)$. The LLM is in fact producing some posterior under *its own implicit prior* over states (whatever its training corpus suggests as the base rate over "tanker incident frequency", "militia attack intensity", and so on). When the LLM's implicit prior is close to the BN's, the prior-squared diagnosis holds quantitatively; when they diverge, the *direction* of the bias is still wrong (the prior is mis-applied either way), but the magnitude of the percentage-point shift in the numerical example is fuzzier than the clean derivation suggests. The fix in Option 1 below is correct regardless of which prior the LLM uses internally, because the likelihood-ratio prompt removes the prior from the LLM's side of the contract entirely. See §C1 *"LLM-implicit-prior leakage"* for the symmetric concern about the new likelihood-ratio prompt.

*Numerical example (with the actual Hormuz prior).* `Tanker_Incidents` has implied prior $\pi = (0.44, 0.36, 0.21)$ over states (`none`, `isolated`, `frequent`) — computed by running variable elimination on the network with no evidence. Translator emits $T = (0.05, 0.15, 0.80)$ — *"this article makes `frequent` very likely."*

pgmpy's computation:
- Unnormalised: $(0.05 \cdot 0.44, \; 0.15 \cdot 0.36, \; 0.80 \cdot 0.21) = (0.022, \; 0.054, \; 0.168)$.
- Sum: $0.244$.
- Normalised posterior: $(0.090, \; 0.221, \; 0.689)$.

**The dashboard shows 69% on `frequent`, not 80%.** The 11-percentage-point gap is the prior-squaring bite — the article carried evidence, but the prior was pulled in twice.

*Empirical confirmation.* The double-counting is reproducible in three lines of pgmpy with an artificial network: build a node $N$ with asymmetric prior $P(s_0) = 0.9, P(s_1) = 0.1$; pass `virtual_evidence` values $(0.8, 0.2)$; observe pgmpy returns $(0.973, 0.027) = $ normalise$(0.8 \cdot 0.9, 0.2 \cdot 0.1)$. If pgmpy treated the input as a posterior, it would have returned $(0.8, 0.2)$ directly. It didn't.

*The fix under Option 1 below.* Re-prompt the LLM for likelihood ratios:
$$\varepsilon_i \;=\; \frac{P(A \mid s_i)}{\max_{i'} P(A \mid s_{i'})}$$

With the max-pinned convention, the best-supported state has $\varepsilon = 1$ and others lie in $(0, 1]$. The max-pinning is an elicitation discipline — pgmpy doesn't care, since proportionality is what matters. pgmpy's computation now gives the right answer:
$$P(s_i \mid V = v) \;\propto\; \varepsilon_i \cdot P(s_i) \;\propto\; P(A \mid s_i) \cdot P(s_i) \;\propto\; P(s_i \mid A)$$

Single multiplication by the prior, exactly as Bayes prescribes.

**Where the prior lives, and what's NOT being changed.**

The "prior" referred to above is whatever the BN computes when asked for the node's marginal with no evidence:

- **Root nodes** ($U_1, U_2, U_3, U_4$): the prior is the root CPT directly. E.g., `CPD_NEGOTIATIONS = [0.20, 0.55, 0.25]` at [src/network.py:100-102](src/network.py#L100-L102) *is* the prior on `US_Iran_Negotiations`.
- **Non-root nodes** (everything else): the prior is **not typed anywhere** — it falls out of marginalising the upstream chain. What lives in the code for these nodes is a *conditional* CPT. `CPD_TANKERS` at [src/network.py:150-164](src/network.py#L150-L164) is $P(\text{Tankers} \mid \text{Militia}, \text{Negotiations})$ — nine columns, one per parent configuration — not a prior. To recover the prior you push the root priors forward through every intervening CPT:
  1. Roots: Regime $= (0.30, 0.50, 0.20)$, Sanctions $= (0.15, 0.55, 0.30)$, Negotiations $= (0.20, 0.55, 0.25)$.
  2. Marginalise through `CPD_MILITIA`: $P(\text{Militia}) = \sum_{r, s} P(\text{Militia} \mid r, s) \cdot P(r) P(s) \approx (0.36, 0.39, 0.25)$.
  3. Marginalise through `CPD_TANKERS`: $P(\text{Tankers}) = \sum_{m, n} P(\text{Tankers} \mid m, n) \cdot P(m) P(n) \approx (0.44, 0.36, 0.21)$.

  The code path is `BNInferenceEngine.get_node_marginal(node)` ([src/inference.py:96-113](src/inference.py#L96-L113)) called with no evidence; pgmpy's VE does the entire chain in a single call.

Two practical consequences:

1. **The prior is a moving target.** "Prior" in the prior-squaring math means *whatever the BN's marginal at $N$ is at the moment the article's likelihood gets multiplied in.* If the analyst has already applied a headline about militia activity, the marginal at Tanker_Incidents that the next article's evidence gets multiplied against is no longer $(0.44, 0.36, 0.21)$ but the post-militia-evidence value — pgmpy recomputes it on every query.
2. **The bug is robust to that.** The numerical magnitude in the example above uses the no-evidence Tanker marginal, but the structural form ($P(A \mid s_i) \cdot P(s_i)^2$ at the Tanker node) holds whatever marginal happens to be in force at query time. The fix in Option 1 below works the same way regardless of upstream evidence state.

**A1 does NOT modify the prior.** The fix touches only:
1. The translator's system prompt ([src/translator.py:134-169](src/translator.py#L134-L169)) — asks for likelihood ratios instead of a sum-to-1 distribution.
2. The validator `_validate_payload` ([src/translator.py:218-294](src/translator.py#L218)) — enforces $\varepsilon \in (0, 1]$ with at least one $\varepsilon = 1.0$ instead of sum-to-1. The open lower bound is deliberate: $\varepsilon = 0$ would assert that the article makes state $s$ *strictly impossible*, propagating as a zero in `virtual_evidence` and zeroing the posterior on that state irrecoverably. Real LLM evidence is never that categorical, so the validator rejects $\varepsilon = 0$ and the prompt instructs the model to use a small positive value (e.g., $0.01$) for "essentially ruled out" rather than $0$.
3. The audit log's `semantics_version` field in D3 — pre-A1 records get `"pre-A1-posterior"`, post-A1 records `"likelihood-ratio"`.

The CPTs in `src/network.py` — root priors, chain CPTs, all of them — stay untouched. The fix is at the **LLM-to-pgmpy interface**, not in the BN. (Re-elicitation of the BN's priors is a separate question, addressed by Plan 4.)

**What to add.** Pick a single semantics and align both ends to it. Three options:

1. *Likelihood-ratio output (recommended).* Prompt the LLM for relative evidence weights `ε_s = P(article | state=s) / max_s' P(article | state=s')`. This is a natural "likelihood" output with a clean reference point: the best-supported state has `ε=1.0` and others are fractions. It maps directly onto pgmpy's virtual-evidence convention without modification.
2. *Posterior output, prior-divided client-side.* Keep the current prompt; before injecting, compute `likelihood(s) = translator(s) / prior(s)` and renormalise. Requires that the engine expose the per-node prior at translation time and assumes the LLM's "prior" is the BN's prior — a non-trivial assumption.
3. *Posterior output, treated as evidence anyway (status quo, documented).* Accept the double-counting as a desirable damping effect and document it as a deliberate choice. Defensible only if the priors are intentionally "sticky" — this should be argued explicitly, not inherited.

**Why it matters.** This is the single most consequential change in the plan. Every node marginal, credible interval, robustness badge, and scenario percentage *computed under translator-injected evidence* flows through this interface — i.e., everything on the dashboard once the analyst has added at least one observation. Until it is settled, no downstream improvement to translator quality is measurable in the inference output.

**Recommendation.** Option 1. Add one paragraph to `docs/model_documentation.md` formalising the contract, update the system prompt, change `_validate_payload` to enforce `0 < ε ≤ 1` with at least one `ε = 1` (open lower bound — see the validator-change item above for the rationale), and update `_virtual_evidence_cpds()` accordingly. Closes M2 and C5.

**Deployment.** Ship A1 as a single atomic change: the system-prompt edit, `_validate_payload` enforcement of the likelihood-ratio shape, and `_virtual_evidence_cpds()` change land together in one PR. No transition window, no dual-semantics support. This matches the deployment shape (single-tenant, one analyst per engagement) where there are no other dashboards depending on the old interface. *Alternatives considered, not implemented: (a) carry both semantics behind a `semantics_version` field in the audit log and let inference dispatch on it during a deprecation window — useful if multiple dashboards consumed the old API, which they don't here; (b) keep the posterior-shaped prompt and divide by the prior client-side — useful if the prompt is hard to change cheaply, which it isn't here.*

**`semantics_version` tagging — timing.** A1 ships at execution slot 1; D3 (the audit log itself) ships at slot 9. The tag therefore has to live somewhere in the interim. The contract is:

- From slot 1 onward, every fresh `TranslatorResult` carries `semantics_version = "likelihood-ratio"`. This is a field on the result object — it does not require D3 to exist.
- The session-scoped observation list in the dashboard already persists `TranslatorResult` records. From slot 1 onward, those records carry the new tag.
- Any pre-A1 records that survive the slot-1 cutover (e.g., notebook-pickled observation lists, exports the analyst has kept) are tagged retrospectively at D3-ingestion time with `semantics_version = "pre-A1-posterior"`. D3's ingest path treats a missing `semantics_version` field as that legacy tag.
- In practice the demo cadence and the single-tenant deployment shape make the surviving pre-A1 corpus very small. The tag exists for completeness, not for a live deprecation.

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

> *Implementation note.* B1 splits into B1a (Article dataclass + paste-with-body + piped-feed inputs + default-per-source-type credibility) at execution slot 4, and B1b (per-source credibility editing with history) at slot 10. B1a sits at slot 4 — ahead of B2 (span-grounded reasoning) and B3 (relevance filter) — so that both operate on full article text rather than the headline alone. B1b sits after D3 (slot 9) because the audit log must be live before the credibility value at translation time can be pinned and recovered. See the execution-order table for the full sequencing.

**What exists now.** The dashboard passes a single headline string to `translate_headline()` at [src/translator.py:462-496](src/translator.py#L462-L496). There is no source, URL, body, dateline, or credibility metadata.

**What to add.** A new `Article` dataclass that carries `{headline, lede, body, source, source_type, url, published_at, language}` and a `translate_article()` entry point that takes it. Source type is a categorical: `wire_service`, `commercial_press`, `state_media`, `analyst_note`, `social_media`, `unknown`. The prompt receives the body (or first ~500 tokens of it) and is instructed to weight the body over the headline when they disagree.

For the demo workflow where an analyst pastes only a headline, the body is left blank and the prompt is told so explicitly. This degrades gracefully — the translator still works on a headline alone but knows it is working without full context.

**Why it matters.** Headlines are written for attention; bodies carry the qualifiers that disambiguate state. "Tanker attacked off Hormuz" could be `isolated` or `frequent` depending on whether the body says "the third such incident this week." Without the body, the translator is guessing.

**Source-credibility weighting.** Each article carries a scalar $w \in [0, 1]$ that discounts the article's likelihood evidence. Sources are addressed by stable identifier (domain, outlet name, or analyst-defined tag) and $w$ is looked up at translation time from a user-editable table.

**Formula — power-likelihood discount.** Apply $w$ as a power on the per-state likelihood ratio, equivalently a multiplicative weight on the log-likelihood:

$$\varepsilon^{\text{weighted}}_i \;=\; \varepsilon_i^{\,w} \qquad\Longleftrightarrow\qquad \log \varepsilon^{\text{weighted}}_i \;=\; w \cdot \log \varepsilon_i$$

Boundary behaviour:

- $w = 1$: full evidence (e.g., a wire-service report).
- $w = 0$: $\varepsilon^0_i = 1$ for every state — no information is injected, equivalent to the article being routed to B3's `relevant=no` path.
- $w \in (0, 1)$: smooth interpolation. The article still nudges the posterior, but proportionally less than a fully credible source would.

The weighting is applied per article (the same $w$ scales every claim and every state from that article) and is composed with C1's ensemble aggregation and B2's per-claim aggregation per the recipe in §C1.

This shape is not ad hoc — it converges on the same form from three literatures:

- **Power likelihood / generalised Bayesian inference** (Bissiri, Holmes & Walker 2016, *JRSS-B* 78(5):1103–1130, *"A general framework for updating belief distributions"*). Replacing $L$ with $L^\eta$, $\eta \in (0, 1]$, is the principled discount on likelihood when the data-generating process is partially trusted; $\eta$ has a formal interpretation as a *learning rate* and the update remains coherent under standard generalised-Bayes axioms.
- **Logarithmic opinion pools** (Genest & Zidek 1986, *Statistical Science* 1(1):114–135, *"Combining probability distributions: A critique and an annotated bibliography"*). Per-source weights on log-likelihoods are the unique aggregation rule satisfying *external Bayesianity* (aggregating-then-updating commutes with updating-then-aggregating). Our $w$ is the log-pool weight.
- **Cooke's classical model** (Cooke 1991, *Experts in Uncertainty*, Oxford UP) — already Plan 4's primary protocol. Cooke weights are calibration-derived exponents on expert likelihoods; using the same shape for source credibility keeps Plans 2 and 4 mathematically aligned.

**Default values per `source_type`** (initial table, analyst-editable per the "living document" workflow below):

| `source_type` | default $w$ | rationale |
|---|---|---|
| `wire_service` | 1.0 | AP, Reuters, AFP — the calibration baseline. |
| `commercial_press` | 0.8 | Major outlets with editorial standards; non-trivial residual bias risk. |
| `analyst_note` | 0.7 | Domain-expert assessments; high signal but not externally verified. |
| `state_media` | 0.3 | Known agenda-driven framing; still carries factual content worth ~30% of a wire. |
| `social_media` | 0.2 | Eyewitness signal sometimes valuable, often noise. |
| `unknown` | 0.5 | Neutral prior pending analyst assignment. |

These are starting values for a fresh deployment; the table below specifies how the analyst evolves them.

The table is a **living document** maintained from inside the dashboard:

- A "Sources" tab lists every source seen in the audit log alongside its current credibility score, the date of the last edit, and the analyst who edited it.
- The analyst can update an existing source's score or pre-populate a new source before its first article is translated.
- Every edit writes a new row to a `source_credibility_history` table — the score in force at translation time is the value most recently committed before that translation's `created_at` timestamp. Past translations are not retroactively rescored.
- A default fallback $w$ per `source_type` (table above) covers unseen sources until the analyst assigns an explicit per-source value.

This makes the audit log fully reproducible (D3 stores the `(source_id, credibility_at_translation_time)` pair on every record) while letting the analyst's view of source quality evolve as new outlets appear and old ones change behaviour.

### B2. Structured reasoning with span-grounding

**What exists now.** Single LLM call: article in, JSON out. The model is free to invent claims that have no textual basis.

**What to add.** A three-step structured pipeline, each step a separate strict-schema LLM call:

1. **Claim extraction.** Input: article. Output: a list of atomic claims, each with `{subject, predicate, object, verbatim_span, confidence}`. `verbatim_span` is a copy-paste from the article body. Claims that don't appear as substrings of the article are rejected.
2. **Per-claim node mapping.** Input: a claim + the node taxonomy. Output: zero or one node assignment with `{node, state, likelihood_ratios, supporting_span_indices, reason}`. A claim that maps to no node is dropped silently.
3. **Per-node aggregation.** Input: all claim-level assignments. Output: one assignment per node, with likelihood ratios produced by combining per-claim ratios **additively in log space** (equivalently, multiplicatively in linear space — the standard independent-evidence combination rule), renormalised once at the end via A1's max-pin convention. The full recipe — including how this composes with C1's self-consistency ensemble across samples and B1's source credibility weight $w$ — is specified once in §C1 below.

   **Independence caveat.** Combining claims multiplicatively assumes they are conditionally independent pieces of evidence about the same node given $N$'s state. Two failure modes break this assumption to different degrees:

   - **Paraphrase duplication** (caught by dedup). Near-paraphrases of the same fact ("tanker hit off Hormuz" / "vessel struck near the strait") would otherwise double-count. **Dedupe at extraction time**, not at aggregation: in step 1, two claims whose `verbatim_span` embeddings have cosine similarity above a threshold (default 0.9) are merged into one before they reach step 3. The `verbatim_span` substring check (claims must be copy-pasted from the article) helps but does not catch paraphrase duplication on its own — the embedding check does.
   - **Distinct claims about the same incident** (residual, not caught). Two claims about the same underlying event with low verbatim-span similarity — "tanker hit by limpet mine" + "vessel taking on water near Larak Island" — remain conditionally dependent given $N$'s state (both are observations of one underlying incident) but pass the dedup filter. The aggregation will overweight them. This is a **known residual limitation** of the per-claim multiplication scheme; the principled fix (per-incident clustering before claim-level aggregation) is deferred. The pragmatic mitigation is that the C1 ensemble's disagreement metric tends to surface high uncertainty on incidents the model is reading multiple ways, which routes them to HITL review (E1).

   For the structural rationale behind requiring verbatim spans in step 1, the canonical references are Bohnet et al. 2022, *Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models*, and Gao et al. 2023, *Enabling Large Language Models to Generate Text with Citations* (EMNLP 2023) — the same "cite-then-claim" discipline that grounds attributed QA.

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

**What to add.** Replace the single call with $N = 5$–$10$ calls at temperature $\approx 0.4$, collect the resulting per-state likelihood vectors, and aggregate them into an empirical distribution.

**Unified aggregation recipe (claims × samples × source credibility).** This is the canonical recipe — the same procedure is referenced from B1 (source credibility) and B2 (per-claim aggregation). Work entirely in **log-likelihood space** and **defer renormalisation to the very end**, so the order of aggregation does not matter and the math has a clean interpretation.

For each node $N$ with states $i$:

1. **Cell.** For each claim $c$ (from B2 step 2) and each sample $s$ (from the $N$ ensemble calls), the structured-reasoning pipeline produces a per-state log-likelihood $\log \varepsilon^{c,s}_i$. **Do not normalise per cell.**

2. **Sample axis — geometric mean across samples, per claim.** For each claim $c$ that appears in at least $N_c$ samples (after applying the minimum-vote rule below):
$$\log \bar\varepsilon^{c}_i \;=\; \frac{1}{N_c} \sum_{s} \log \varepsilon^{c,s}_i$$
This is the within-model variance reduction — it averages out sampling noise.

3. **Claim axis — sum across claims (independent evidence combination).**
$$\log \varepsilon^{\text{node}}_i \;=\; \sum_{c} \log \bar\varepsilon^{c}_i$$
Linear in log space = multiplicative in linear space = independent-evidence Bayes combination. (Independence is enforced upstream by the dedupe step in B2.1; see B2 for the caveat.)

4. **Source credibility weight — power discount on the article.** Apply B1's $w \in [0, 1]$ as a single multiplicative weight on the aggregated log-likelihood:
$$\log \varepsilon^{\text{weighted}}_i \;=\; w \cdot \log \varepsilon^{\text{node}}_i$$
Equivalent to $\varepsilon \leftarrow \varepsilon^{w}$ on the linear scale. See B1 for the literature anchoring this form (Bissiri/Holmes/Walker 2016; Genest/Zidek 1986; Cooke 1991).

5. **Renormalise once, at the end — A1's max-pin convention.**
$$\log \varepsilon^{\text{final}}_i \;=\; \log \varepsilon^{\text{weighted}}_i \;-\; \max_{i'} \log \varepsilon^{\text{weighted}}_{i'}$$
Then exponentiate to get $\varepsilon^{\text{final}}_i \in (0, 1]$ with $\max_{i'} \varepsilon^{\text{final}}_{i'} = 1$. This is the vector fed to pgmpy's `virtual_evidence`.

**Why this order, and why renormalisation lives at the end.** Step 2 collapses the sample axis *per claim* with a normalised average ($\frac{1}{N_c} \sum_s$), and step 3 collapses the claim axis with an unnormalised sum ($\sum_c$). The two cannot, in general, be swapped: because $N_c$ varies across claims (the minimum-vote rule allows $N_c \in [\lceil N/2 \rceil, N]$), reversing the axes would re-weight every claim by a factor of $N_c/N$. The chosen order is the principled one: it gives each claim equal voice in step 3 regardless of how many samples happened to emit it, while sample-axis noise is collapsed first within the claim where it actually arose. Step 4 (scalar $w$ on the aggregated log-likelihood) and step 5 (subtract the max) are both monotone scalar operations and commute with each other and with whatever quantity step 3 produces. Renormalisation lives at the very end because that is where path-dependence would otherwise enter — by deferring it, intermediate quantities stay on a comparable additive log scale and the recipe is well-defined and reproducible from the per-cell $\log \varepsilon^{c,s}_i$ inputs alone.

**Disagreement metric for E1's HITL trigger.** Compute the per-state standard deviation of $\log \varepsilon^{c,s}_i$ across samples $s$, **per claim $c$, before step 2 collapses it**:
$$\sigma^{c}_i \;=\; \mathrm{StdDev}_{s}\bigl(\log \varepsilon^{c,s}_i\bigr)$$
The node-level disagreement score is $\max_{c, i} \sigma^{c}_i$. This is what E1 thresholds against. Do not compute disagreement on the post-aggregated quantity — by step 3 the within-model variance signal has been collapsed.

**Minimum-vote rule (preserved).** A claim $c$ that appears in fewer than $N/2$ of the $N$ samples is dropped before step 2 as *"translator was inconsistent about whether to assign this claim at all."* Similarly, a node that ends up with zero surviving claims is not emitted as an assignment.

**Note on $N_c$ and confidence.** The geometric mean in step 2 averages over $N_c$ samples — the count of samples that actually emitted claim $c$ — which can be anywhere from $\lceil N/2 \rceil$ up to $N$ after the minimum-vote filter. The aggregated $\log \bar\varepsilon^c_i$ is on the same magnitude scale regardless of $N_c$, but the *statistical confidence* in the average differs: a claim with $N_c = 3$ has weaker estimator confidence than a claim with $N_c = 5$. C1 surfaces this only via the disagreement metric (high $\sigma^c_i$ on a low-$N_c$ claim is the principled trigger for HITL review). $N_c$ itself is **not** propagated as an explicit confidence weight into the aggregation — this is a deliberate simplification that keeps the per-claim contribution to step 3 independent of how many samples happened to emit it, and reconsidered only if calibration plots in D2 show systematic mis-coverage on low-$N_c$ claims.

**Why geometric (not arithmetic) mean across samples.** Likelihood ratios compose multiplicatively, so under A1's likelihood-ratio output the arithmetic mean of $\log \varepsilon$ values (i.e., geometric mean in linear space) is the principled aggregator. It is also robust to single-sample outliers: if four samples give $\varepsilon_{\text{severe}} = 2$ and one gives $\varepsilon_{\text{severe}} = 10$, arithmetic mean is $3.6$ (pulled by the outlier) while geometric mean is $\approx 2.7$ (closer to the typical sample).

The per-sample raw outputs, the per-claim disagreement values, the source credibility $w$ applied, and the final aggregated likelihood vector all live in the audit log (D3) so reproducibility is preserved end-to-end.

**Why it matters.** This is the proper answer to *"where does `state_probs` come from."* Instead of trusting the LLM's hand-rolled posterior, you measure the LLM's own consistency and let that drive the soft-evidence width. Disagreement across samples becomes honest uncertainty rather than synthetic confidence.

**What this captures, and what it doesn't.** C1 measures **within-model sampling variance** — *"how consistent is the model with itself when re-rolled?"* This catches:
- Ambiguous articles where the model could read the text multiple ways.
- Edge cases where the model is genuinely on the fence between two states.

C1 does **not** catch:
- **Model misspecification** — the model is systematically and *consistently* wrong (all $N$ samples say `frequent` for an article actually about `isolated`). C1 reports low uncertainty in this case, falsely. Backstopped by C2 (multi-model cross-check).
- **Distribution shift** — the article is unlike anything in the model's training distribution; the model produces plausible-looking nonsense at low variance. Partly backstopped by E2 (retrieval against past analyst-approved translations).
- **Absolute miscalibration** — the variance estimate is fine but the confidence numbers are systematically off across the corpus. Backstopped by D2 (calibration plots).
- **LLM-implicit-prior leakage.** A1's likelihood-ratio prompt asks the model for $\varepsilon_s = P(\text{article} \mid s) / \max_{s'} P(\text{article} \mid s')$, but there is no mechanism enforcing that the LLM actually produces a likelihood-shaped quantity rather than something tinted by its own implicit prior over states (e.g., training-corpus base rates over "tanker incident frequency"). The same risk applies symmetrically in reverse to the diagnosis of the current posterior-shaped prompt (the prior-squaring math assumes the LLM's "prior" *is* the BN's prior, which is unverifiable). C1's ensemble does not break this — all $N$ samples are drawn from the same model and inherit the same implicit prior. Partly diagnosable via D2's calibration plots (systematic bias against the gold set shows up as miscalibration) and partly backstopped by C2 (an LLM with a different implicit prior gives a comparison reference).

**References.**

*Self-consistency family (the sample-and-aggregate idea).* Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, & Zhou (2022), *"Self-Consistency Improves Chain of Thought Reasoning in Language Models"* (ICLR 2023) — the original, for reasoning tasks (sample $N$ chains, majority vote). Generalises to sampling-based LLM uncertainty quantification: Kuhn, Gal, & Farquhar (2023), *"Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation"* (ICLR 2023), for the clustering-based extension; Manakul, Liusie, & Gales (2023), *"SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection"* (EMNLP 2023), for the factual-consistency variant. C1 is the practical baseline in this family.

*Log-space aggregation (the recipe's mathematical grounding).* Genest & Zidek (1986), *Statistical Science* 1(1):114–135 — logarithmic opinion pools as the unique externally-Bayesian aggregator. Bissiri, Holmes & Walker (2016), *JRSS-B* 78(5):1103–1130 — power likelihood / generalised Bayes (the formal basis for B1's $\varepsilon^w$ credibility weighting, applied here in step 4). Hoeting, Madigan, Raftery & Volinsky (1999), *Statistical Science* 14(4):382–417 — BMA-style multiplicative combination over independent evidence, the rationale for step 3.

**Complements worth knowing.**
- **Verbalised confidence** (Lin, Hilton, & Evans 2022, *"Teaching Models to Express Their Uncertainty in Words"*, TMLR). Ask the LLM to rate its own confidence on a 1–10 scale alongside the assignment. Single call, ~50 extra output tokens, often surprisingly well-calibrated for frontier models including Sonnet 4.5+. Recommended as a cheap add-on field per sample; cross-check against C1's disagreement metric in D2's calibration plots.
- **Logit-based confidence.** Use API-exposed token probabilities for the answer token. Cheaper than ensemble and often better-calibrated, but currently not actionable under the Claude-only posture (Claude's logprob exposure is limited). Bookmark for the OpenAI cross-check in C2.
- **Conformal prediction** (Quach, Fisch, Schuster, Yala, Sohn, Jaakkola, & Barzilay 2024, *"Conformal Language Modeling"*, ICLR 2024). Provides finite-sample coverage guarantees rather than confidence estimates. Heavier engineering; right tool if/when stakeholder decisions require a defensible coverage bound.

**Cost note.** N× LLM spend per article. At demo cadence this is negligible (~$0.05/article on Sonnet 4.5 at N=5). At production cadence it scales linearly; below 1000 articles/day the bill is still small.

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

**What each size unlocks.** The metrics above do not all light up at the same corpus size — calibration plots and per-node F1 are statistically meaningless on a 30-record set where most nodes appear ≤2 times. The expectations are:

- **30 records (slot 3 MVP).** Gates contract regressions only: does A1 still validate? does A2 reject malformed payloads? does the overall pipeline return without crashing on a representative slice? Useful for go/no-go decisions on A1/A2 ships. Calibration and per-node metrics are *not* reliable at this size — any reported numbers are illustrative, not decisional.
- **50 records (D1 CI-gate threshold, slot 8).** Each node appears at least twice. Node-recall and node-precision become meaningful at the *aggregate* level; per-node F1 is still noisy but directionally usable. CI gate enforced at "aggregate F1 minus tolerance" rather than per-node.
- **100–200 records.** Per-node F1 becomes reliable. Calibration plots have enough mass per bin to detect systematic miscalibration. This is the size at which the "where do the numbers come from" trust question is genuinely defensible.
- **300–500 records (six-month target).** Per-state accuracy conditional on node-match becomes reliable. Abstention precision/recall are well-resolved. Brier score has enough mass to detect drift between model versions.

The dashboard "translator accuracy" badge surfaces only the metrics whose corpus size supports them; lower-confidence numbers appear with explicit `n =` annotations or are suppressed until threshold.

**Implementation note.** Labelling cost is real once the workflow is in place. The golden set is the single most expensive artefact in this plan in human time.

### D3. Provenance and audit log

**What exists now.** `TranslatorResult` carries `{headline, assignments, rationale, model, provider, raw_response}` ([src/translator.py:51-67](src/translator.py#L51-L67)). The dashboard logs assignments into the observation list but does not persist source URL, prompt version, raw-response hash, or analyst-approval state.

**What to add.** Extend `TranslatorResult` to carry `{article_url, source, source_credibility, prompt_version, model, model_version, response_hash, temperature, ensemble_size, sample_disagreement, created_at, relevance, analyst_state, analyst_id, analyst_correction, body_retention}`. Persist to a structured log keyed by `response_hash`. The dashboard observation log becomes a thin view over this store.

**Storage substrate — sqlite primary, parquet for analytics.** The live audit log is a single-file **sqlite** database per deployment: ACID writes, transactional analyst-state edits, easy to ship inside the on-premise stack, and well-supported by the Python ecosystem. A nightly export to **parquet** sits alongside it for analytical workloads (calibration plots, retrieval-corpus rebuilds, batch sensitivity studies) where columnar reads dominate. sqlite is the source of truth; parquet is a derived artefact and can be rebuilt at any time from sqlite. This split matches the deployment shape (single-tenant, per-engagement) — there is no multi-writer SaaS workload that would justify a heavier substrate.

**Article-body retention policy — store the body by default.** The body is the load-bearing audit artefact. News URLs go dead, get paywalled, and get silently re-edited within months; re-fetch is not a reliable reproducibility mechanism for a stakeholder-facing audit trail in regulatory or intelligence contexts. For each translation the audit log persists `article_url`, `source`, `headline`, `body` (verbatim, as the translator saw it), `body_sha256`, and `body_length`. The single-tenant on-premise deployment shape makes per-deployment content-licensing storage concerns manageable — they are answered once per engagement at deployment-contract time, not piecemeal in the audit code.

**Hash-only retention as opt-in.** For content the analyst flags as licensed or otherwise non-redistributable, set `body_retention = "hash_only"` on the per-source registry entry; subsequent translations from that source persist `body_sha256` and `body_length` only, with `body = NULL`. The reproducibility contract for these records is downgraded to "re-fetch from URL, verify hash matches" and the audit UI displays the hash-only marker. The default (`body_retention = "stored"`) applies to every source absent an explicit opt-in.

Reproducibility contract: given `(article_url, body_sha256, prompt_version, model, model_version, temperature, ensemble_size)` and the stored `body`, re-running the translator yields the same `response_hash` up to LLM sampling noise (the noise envelope is documented from the C1 self-consistency measurements). For hash-only records the contract is the same but conditional on the re-fetched body's hash matching `body_sha256`.

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
| 4 | B1a: Article-level input | Reasoning | (2) | Article dataclass with all three input pathways supported (paste-only, paste-with-body, piped feed); default-per-source-type credibility weights only. Doubles the available signal at the input layer, and lands **before** B2/B3 so that span-grounding and the relevance pre-filter both work on full article text rather than just the headline. Does not yet require the audit/pinning infrastructure (B1b is split out to slot 10 for that). |
| 5 | B3: Relevance filter and abstention | Reasoning | (4) | Cheap, high-impact. Adds an honest "not relevant" path. The embedding pre-filter is meaningfully more accurate against the full article (B1a, slot 4) than against the headline alone. |
| 6 | B2: Span-grounded structured reasoning | Reasoning | (2 partial, 5) | The largest single jump in translator trustworthiness. Eliminates hallucination at the assignment level by forcing every claim to cite verbatim source text — which only has meaningful surface area once B1a (slot 4) is in place. Depends on A2 (schema hardening) being clean. |
| 7 | C1: Self-consistency ensemble | Uncertainty | (3) | Replaces the LLM's hand-rolled `state_probs` with an empirically measured distribution. Depends on A2 being stable so the ensemble aggregator can rely on a canonical shape, and on B2 (slot 6) so the per-claim aggregation axis exists. |
| 8 | D1: Prompt as versioned artefact | Governance | (6) | Operational hardening. Trivial to build once prompts are externalised; the value is the CI gate that prevents silent regressions. |
| 9 | D3: Provenance audit log | Governance | (8) | Reproducibility contract. Prerequisite for E2 (RAG retrieval needs a structured store), for B1b (per-source credibility pinning), and for any compliance review. |
| 10 | B1b: Per-source credibility with history | Reasoning | (2) | The Sources tab and `source_credibility_history` table. Depends on D3 (slot 9) being live so translations can be pinned to the credibility value in force at their `created_at`. |
| 11 | C2: Multi-model cross-check | Uncertainty | (3) | Compounds with C1 — catches systematic per-model biases. Optional toggle; default on for analyst workflow, off for batch ingest. |
| 12 | E1: HITL review queue (threshold-triggered) | Operations | (4, 7) | The bridge between an imperfect model and a defensible workflow. Threshold-triggered: only borderline translations enter the queue; the rest auto-approve. Compounds with D2: analyst corrections accrue into the golden set. |
| 13 | E2: Retrieval-augmented translation | Operations | (3, 6) | Reach item. Depends on D3 + E1 being populated. Couples directly to E1 in [docs/bn_app_next_steps.md](docs/bn_app_next_steps.md) — they should share one news-memory layer. |

**Minimum viable correctness baseline.** Items 1–3 (A1, A2, D2-MVP) fix the worst interface bugs and give the team a measurement loop. Everything after that compounds against that loop.

**Largest single quality jump.** Items 6–7 (B2 span-grounded reasoning, C1 self-consistency ensemble) together deliver the biggest stakeholder-visible improvement in translator output and calibration — with item 4 (B1a article-level input) as the prerequisite that makes the body text available for both to operate on.

**Institutional infrastructure.** Items 8–13 build the layer that distinguishes a tool from a script: versioning, audit, review, memory.

---

## Design Decisions

The decisions below are resolved.

1. **Likelihood semantics (A1) — Decided: likelihood-ratio output.** The prompt asks the LLM for $\varepsilon_s = P(\text{article} \mid s) / \max_{s'} P(\text{article} \mid s')$. The best-supported state pins at $\varepsilon = 1.0$; others are fractions in $(0, 1]$. Maps directly onto pgmpy's virtual-evidence convention without modification (see the math derivation in §A1 above). Closes M2/C5 from the review. **The fix does not modify the BN's priors** — only the LLM prompt, the validator, and the audit log's `semantics_version` field.
2. **A1 deployment shape — Decided: atomic single PR.** Prompt change, validator change, and `_virtual_evidence_cpds()` change land together. No transition window, no dual-semantics support. Justified by the single-tenant deployment shape (no other dashboards consume the old API). Audit-log records pre-dating A1 are tagged retrospectively with `semantics_version = "pre-A1-posterior"`.
3. **Input pathway (B1) — Decided: all three supported, analyst chooses per article.** Paste-only headline (current behaviour preserved), paste-with-body, and a piped feed (RSS/GDELT) are all valid inputs. The `Article` dataclass tolerates missing `body`, `url`, `published_at`; the structured-reasoning prompt (B2) is told explicitly which fields are present so it can downgrade confidence when working from a headline alone.
4. **Golden-set authorship (D2) — Decided: single-author start, expand later.** Francesco labels the v0 set (30–50 records) alone. As resources expand, additional annotators are folded in; the doc schema already accommodates a per-record `annotator` field and inter-annotator agreement metrics can be added once N ≥ 2.
5. **Compute budget per article — Decided: accept demo-cadence cost for now.** Self-consistency at N=5 + 3-step structured reasoning multiplies per-article LLM cost ~15×. Acceptable at current demo cadence (≤10 articles/day). Revisit when daily volume passes 100 articles or when a high-throughput batch mode is proposed.
6. **Source-credibility table (B1) — Decided: living document, user-maintained from inside the dashboard.** The analyst assigns and updates per-source credibility scores $w \in [0, 1]$ directly in a "Sources" tab. Defaults per `source_type` cover unseen sources (see B1 table for initial values). Every edit appends to a `source_credibility_history` table; past translations are pinned to the score in force at their `created_at` timestamp and are not retroactively rescored. The audit log (D3) carries `(source_id, credibility_at_translation_time)` on every record so reproducibility is preserved across edits.
7. **Source-credibility weighting formula (B1) — Decided: power-likelihood discount.** $\varepsilon^{\text{weighted}}_i = \varepsilon_i^{\,w}$, equivalently $\log \varepsilon^{\text{weighted}}_i = w \cdot \log \varepsilon_i$. Boundary behaviour: $w=1$ full evidence, $w=0$ no information, smooth interpolation in between. Grounded in three converging literatures: power likelihood / generalised Bayesian inference (Bissiri, Holmes & Walker 2016), logarithmic opinion pools (Genest & Zidek 1986), Cooke's classical model (Cooke 1991 — Plan 4's primary protocol). See §B1 for the full literature anchor.
8. **C1 / B2 aggregation order — Decided: log-space throughout, renormalise once at the end.** The canonical recipe is specified in §C1 and referenced from §B1 and §B2. Per-node: (1) collect $\log \varepsilon^{c,s}_i$ per claim $c$, sample $s$, state $i$; (2) geometric mean across samples per claim; (3) sum across claims per node (independent-evidence combination); (4) apply source credibility weight $w$; (5) renormalise via A1's max-pin convention. The order matters: step 2 is a normalised average and step 3 is an unnormalised sum, so under variable $N_c$ they do not commute. The chosen order gives each claim equal voice in step 3 regardless of how many samples emitted it. Renormalisation is deferred to step 5 so that intermediate quantities stay on a comparable additive log scale and the recipe is fully reproducible from the per-cell $\log \varepsilon^{c,s}_i$ inputs. Disagreement metric is the per-state standard deviation of $\log \varepsilon^{c,s}_i$ across samples, computed *per claim before step 2 collapses it* (max over claims and states yields the node-level score that gates E1 HITL routing).
9. **Claim deduplication (B2) — Decided: enforce at extraction, not aggregation.** Two claims whose `verbatim_span` embeddings have cosine similarity above a threshold (default 0.9) are merged into one before reaching the aggregation step. This is the mitigation for the independent-evidence assumption in step 3 of the aggregation recipe — paraphrase double-counting is prevented upstream rather than corrected downstream. Note that paraphrase dedup does **not** resolve the deeper conditional-dependence problem (two distinct claims describing the same incident — "tanker hit" + "vessel sank" — remain conditionally dependent given $S$ even after dedup); see §B2 for the residual-limitation note.
10. **D3 article-body retention — Decided: store the body by default.** The audit log persists `article_url`, `source`, `headline`, `body_sha256`, `body_length`, **and the body itself**. The body is the load-bearing audit artefact: news URLs go dead, get paywalled, and get silently re-edited within months, so re-fetch is not a reliable reproducibility mechanism for a stakeholder-facing audit trail. The single-tenant on-premise deployment shape makes content-licensing storage concerns manageable per deployment. Hash-only retention is available as an **opt-in** per-source flag for content the analyst flags as licensed or otherwise non-redistributable; in that mode the audit record carries a `body_retention = "hash_only"` marker and a re-fetch contract instead of the body. See §D3.
11. **HITL operating model (E1) — Decided: threshold-triggered only.** The default flow auto-approves translations that clear the confidence and cross-model-agreement thresholds. Only borderline cases enter the analyst review queue. This is the lighter-weight UX shape and is consistent with the demo cadence in (5). Revisit if analyst workflow expands to always-on triage.
