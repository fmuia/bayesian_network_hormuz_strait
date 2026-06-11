# Plan 1 — Latent Regime Reframing (Hormuz instance)

> **Status.** Conceptual decision ✅ resolved. Framework write-up ✅ shipped (`docs/scenario_bn_framework.md`). Engineering implementation ✅ shipped 2026-06-05 (`src/network.py`, `src/cpt_data.py`, `src/inference.py`, `src/sensitivity.py`; `scripts/derive_latent_regime_anchors.py`; tests in `tests/test_latent_regime.py`). Both topologies are kept side-by-side (labelling remains the default) for the comparison; the latent-regime↔labelling comparison is in `docs/01_latent_regime_comparison.md` and `notebooks/latent_regime_comparison.ipynb`. **Correction:** M7 is *not* closed by the reframe — the point-vs-resample gap is a small Jensen artefact present in both topologies (see comparison §M7); M1 is closed.
>
> **Position in the sequencing.** First plan in the programme. The **conceptual decision** is settled and underlies the specification of Plans 2–5. The **framework write-up** (see B.2) and the **engineering implementation** are the outstanding work. Plan 1's engineering ships before Plan 2 begins and has no engineering prerequisites — it edits `src/network.py`, `src/cpt_data.py`, and `src/inference.py` directly on the existing pgmpy code path. No `NetworkSpec`, no backend abstraction. Plan 3 later lifts the work into the declarative `NetworkSpec` / dual-backend architecture and adds PyMC-native support in Phase 2. The framework write-up has no engineering prerequisites and can proceed in parallel.
>
> **Foundational reference.** This plan applies the *scenario-as-latent BN framework* to the Hormuz network. The framework — five node categories ($\mathcal X, \mathcal M, \mathcal O, S, \mathcal D$) and six edge rules (R1–R6) — is specified in [docs/scenario_bn_framework.md](scenario_bn_framework.md). This plan is the Hormuz **instance** of the framework, not a re-derivation of it. Sections A.1–A.3 below assume framework fluency at the level of that document.
>
> **Related docs.** `docs/master_plan.md` §4 is the in-tree registry of finding IDs and includes M1 / M7 (the findings this plan closes). `docs/03_pymc_integration_plan.md` provides the backend substrate the implementation runs on. Cross-cutting touches in `docs/02_translator_robustification.md` (Plan 2 A1), `docs/04_elicitation_tool_plan.md` (Plan 4 Layer 2), and `docs/05_dashboard_ui_plan.md` (Plan 5 C4) are described in Section C below.
>
> **Status legend.** ⬜ not started · ⏳ in progress · ✅ shipped (with date).

## Executive Summary

The current Hormuz Bayesian network treats `Scenario` as a leaf node with three intermediate-outcome parents (`Energy_Infrastructure_Damage`, `Conflict_Duration`, `Diplomatic_Resolution_Path`) — written $(D, T, P)$ in the math notes — via a 27-column CPT $P(S \mid D, T, P)$. The dashboard review (finding M1) showed that this CPT is mathematically a softmax-like *labelling function* of the three parents, not a generative probabilistic model. Scenario probabilities reported by the dashboard are therefore the expectation of a labelling function under the joint posterior of $(D, T, P)$ — a *coarsening* of the outcome posterior (a legitimate posterior over $S$, but a derived view of the outcomes), not a posterior over a latent regime that carries its own prior and emission likelihoods. The distinction is expressiveness, not correctness; §A.2.3 develops it in full.

The fix is to **instantiate the scenario-as-latent BN framework** (the labelling-CPT trap is the framework's prototypical failure mode; see [framework §1](scenario_bn_framework.md#L11) and the §10 pitfalls list). For Hormuz, this means:

1. **Reverse the $S$-arrows.** $S$ becomes an internal latent node generating $\{D, T, P\}$ as emissions (restoring edge rule R4).
2. **Give $S$ context-summary parents.** $\text{Pa}(S) = \{M, C\}$ — the downstream-most layer of Hormuz mediators (satisfying R5a/R5b). This is what makes upstream evidence (militia attacks, sanctions, tankers, military posture, closure status) propagate to the regime posterior via active chain paths rather than being blocked by colliders at $\{D, T, P\}$.
3. **Document one residual blind spot.** Evidence on $U_3$ (Third_Party_Mediation) alone does not update $P(S)$ in v1; mediation news propagates to $S$ only when accompanied by other diplomatic-channel evidence. Triage decision is to accept and document, with $U_3$ as a candidate additional parent in Plan 4 Layer 2 elicitation.

The plan has three sections of substance. **Section A** documents the conceptual decision, framed in the language of the scenario-as-latent BN framework, with Hormuz as the worked instance. **Section B** specifies the engineering implementation that delivers the reframed network — including the framework write-up itself as a first-class deliverable. **Section C** enumerates the cross-cutting touchpoints with Plans 2–5.

---

## Section A — The conceptual decision (Hormuz as a framework instance)

### A.1 What this plan instantiates

The plan applies the **scenario-as-latent BN framework** to the Hormuz network. The framework distinguishes five node categories and governs their connections via six edge rules. The full definitions are in [docs/scenario_bn_framework.md](scenario_bn_framework.md); this section presumes familiarity. As a one-line gloss:

- $\mathcal X$ — exogenous drivers (roots).
- $\mathcal M$ — process mediators (causal chain between $\mathcal X$ and $\mathcal O$).
- $\mathcal O$ — definitional outcomes (the joint configuration that *defines* which scenario is in force).
- $S$ — latent scenario (the client's named scenarios as a categorical variable).
- $\mathcal D$ — downstream readouts (observable consequences of $\mathcal O$).

The six edge rules R1–R6 are detailed in framework §3. The two that this plan relies on most heavily are:

- **R4** — each $O \in \mathcal O$ has $S$ as a parent (plus its causal-upstream parents). This makes $S$ a generative latent regime.
- **R5** — $\text{Pa}(S) \subseteq \mathcal M \cup \mathcal X$, chosen so that every upstream node has an active chain path to $S$ that does not pass through an $\mathcal O$-collider (R5a), and so that $\text{Pa}(S)$ is the downstream-most mediator layer satisfying R5a (R5b).

The Hormuz partition is:

| Category | Hormuz nodes |
|---|---|
| $\mathcal X$ | $U_1$ Negotiations, $U_2$ Regime Stability, $U_3$ Mediation, $U_4$ Sanctions |
| $\mathcal M$ | $A$ Militia, $K$ Tankers, $M$ Military, $C$ Strait Closed |
| $\mathcal O$ | $D$ Damage, $T$ Duration, $P$ Diplomatic Path |
| $S$ | Scenario: $\{$ Stress_Mitigates, Prolonged_Conflict, Severe_Closure $\}$ |
| $\mathcal D$ | $O$ Oil Price Regime |

This partition is unambiguous: each node passes exactly one of the inclusion tests in framework §2, and the scenario narratives at [src/network.py:364-377](../src/network.py#L364-L377) read as conjunctions over $(D, T, P)$ — confirming $\mathcal O$ via the §5 step 2 sufficiency test.

### A.2 Why the original Hormuz topology is broken — finding M1, in framework language

Finding M1 — *"Scenario as classifier"* in master-plan §4 — is precisely the **labelling-CPT trap** of framework §10. The original Hormuz network has edges $\{D, T, P\} \to S$: $S$ is a leaf, with $\mathcal O$ as its parents. This is a direct **R4 violation**: under R4, arrows must run $S \to O$, not the reverse.

The structural fingerprint of the violation is visible in the **entropy profile** of the original CPT $P(S \mid D, T, P)$ — the 27-column table at [src/network.py:326-361](../src/network.py#L326-L361). On a 3-state distribution, Shannon entropy ranges in $[0, \log_2 3] \approx [0, 1.585]$ bits (§A.2.2 spells out the computation). Computing $H$ for all 27 columns of the live table (verified against the code, not quoted) gives a profile that is decisive only at the two **extreme** corners and diffuse everywhere else:

- **Sharp only at the extreme corners.** The two columns where all three outcomes agree on an extreme regime are near one-hot: `none / short / open` (the Stress fingerprint, $[0.94, 0.05, 0.01]$) at $H = 0.37$, and `severe / long / closed` (the Severe fingerprint, $[0.01, 0.09, 0.90]$) at $H = 0.52$. The next two columns — both strongly Stress-leaning $[0.85, 0.13, 0.02]$ — already sit at $H = 0.70$.
- **Diffuse everywhere else.** The remaining 23 columns are spread: 20 of the 27 sit at $H \ge 1.1$ bits, and the mean over all columns is $1.16$. The most ambiguous approach the ceiling — `moderate / medium / narrowing` ($H = 1.37$), `severe / short / open` ($H = 1.54$).
- **The middle regime is never sharply identified.** No column points cleanly to Prolonged_Conflict: even its best-supporting configurations — e.g. `moderate / long / closed` $= [0.03, 0.55, 0.42]$ — land at $H \approx 1.15$, because severe-damage mass bleeds across the boundary.

This profile is the diagnostic signature of a **soft classifier** — an argmax-with-fuzzy-boundaries over the 27 outcome cells — not a generative emission. Note that it is a *continuum* from $0.37$ to $1.54$, not a clean two-cluster split with an empty middle; the tell is the combination of razor-sharp peaks at the two hand-placed corners with a broad high-entropy mass everywhere else, including the model's own middle category. A genuine emission family $P(O \mid S, \ldots)$ would not produce this table at all — there would be no $P(S \mid \cdots)$ column to inspect — and where a real regime modulates an outcome it shifts the distribution by a consistent amount rather than collapsing to near-certainty at two corners and spreading to near-uniform between them. The structural reason this table can only ever be a classifier, never a generative model, is developed in §A.2.3.

Finding M7 — *"Resample-mean vs point-estimate"* — is closed in practical terms by the reframe as a side-effect. The 1–3pp gap between the dashboard's two computation paths comes mostly from the non-linearity of the labelling CPT: point-estimate inference computes $\text{labelling}(E[D, T, P])$, while the resample-mean computes $E_\theta[E[\text{labelling}(D, T, P) \mid \theta]]$, and these differ whenever labelling is non-linear (sharply, at the corners). Once $P(S \mid E)$ becomes a genuine Bayes-rule posterior over a latent regime variable, the labelling step disappears and the gap shrinks to a much smaller residual (~0.1–0.5pp) from general Bayesian non-linearity — the Jensen's-inequality gap between the point-estimate-of-posterior and the posterior-over-point-estimates. That residual is below the precision at which the dashboard would display percentages, so M7 closes without further Plan 5 UI work.

#### A.2.1 Reading the CPT: columns, hard and soft

The diagnostic above, and the argument in §A.2.3, both describe *columns* of $P(S \mid D, T, P)$ as "hard" or "soft." Precisely:

The CPT is stored (pgmpy convention) as a matrix whose **rows** are the three states of $S$ (Stress_Mitigates, Prolonged_Conflict, Severe_Closure) and whose **columns** are the parent configurations — one per combination of $(D, T, P)$. With $3 \times 3 \times 3$ that is 27 columns. Each column is therefore a single conditional distribution $P(S \mid D{=}d, T{=}t, P{=}p)$: a 3-vector summing to 1, the scenario distribution *for that one outcome combination*.

```
                    (none,short,open)  (severe,short,open)   (severe,long,closed)
Stress_Mitigates         0.94                0.30                    0.01
Prolonged_Conflict       0.05                0.45                    0.09
Severe_Closure           0.01                0.25                    0.90
                      └─ hard col ─┘      └─ soft col ─┘          └─ hard col ─┘
```

- A **hard** column is sharply peaked (near one-hot): the outcome combination points cleanly to one scenario. `(none, short, open)` puts $0.94$ on Stress.
- A **soft** column is spread out (near-uniform): the three outcome variables *disagree* about the scenario, so none dominates. `(severe, short, open)` — severe damage says "Severe," but short duration and an open path say "Stress" — splits roughly evenly.

"Hard/soft" is just low-entropy (peaked) versus high-entropy (flat), measured per column.

#### A.2.2 How column entropy is computed

"Entropy" here is Shannon entropy in bits, computed on one column's 3-vector $p = (p_{\text{Stress}}, p_{\text{Prolonged}}, p_{\text{Severe}})$:

$$H(p) = -\sum_i p_i \log_2 p_i \qquad \text{(convention } 0\log_2 0 = 0\text{)}.$$

A common shortcut is to remember $\log_2 k$ — but that is the *maximum* entropy (achieved only by the uniform distribution), not the general formula. The actual value depends on how the mass is spread.

**Worked example — the `(none, short, open)` column** $p = (0.94, 0.05, 0.01)$:

$$H = -\big(0.94 \log_2 0.94 + 0.05 \log_2 0.05 + 0.01 \log_2 0.01\big) = -\big(0.94(-0.089) + 0.05(-4.322) + 0.01(-6.644)\big) = 0.084 + 0.216 + 0.066 = 0.37 \text{ bits}.$$

Low — almost all mass on one state. A "hard" / decided column.

**Two reference points** anchor the scale:

- **Uniform** $p = (\tfrac13, \tfrac13, \tfrac13)$: each $\log_2 \tfrac13 = -1.585$, so $H = \log_2 3 = 1.585$ bits — the maximum, and the source of the $\log_2 k$ shortcut.
- **One-hot** $p = (1, 0, 0)$: $H = 0$ bits — total certainty.

So a 3-state distribution always has $H \in [0, 1.585]$. Entropy measures how *spread* the mass is, not how many states exist: the same three states give $H = 0$ when peaked and $H = 1.585$ when flat.

### A.2.3 Why the reversal is justified: one structural fact, three consequences

> **Map of the rest of §A.2.** §A.2.3 (here) makes the *structural* case for inverting the arrows — one fact, three consequences. §A.2.4 states the *semantic* price (scenarios as modal signatures) and resolves the identifiability cost. §A.2.5–A.2.6 show the inverted model *operating* — on a hard observation, then on soft translator evidence. §A.2.5 deliberately re-derives this section's Bayes-factor contrast as concrete mechanics: reinforcement for the reviewer, not a second argument.

§A.2 shows the current CPT *behaves like* a classifier. This section explains *why* the labelling topology can never deliver what the brief asks for — and states the argument honestly. The honest claim is **not** "the labelling model produces no posterior over $S$." It does: $P(S \mid E)$ is well-defined for any evidence $E$, and the dashboard already computes it. The honest claim is about **expressiveness** — the labelling topology cannot express two objects this product needs: an independently-chosen regime prior, and per-regime evidence likelihoods (clean Bayes factors). The whole argument rests on one structural fact, and three consequences flow from it.

**The one structural fact.** In the labelling DAG, $S$ is a **leaf** whose only parents are $(D, T, P)$. So $S$'s Markov blanket is exactly $\{D, T, P\}$ — no children, no co-parents — and $S$ is conditionally independent of all other evidence once the outcomes are known:

$$S \perp E \mid (D, T, P) \qquad \text{for any evidence } E \text{ placed elsewhere in the network.}$$

This is exact and structural: it holds no matter how sharp or soft the CPT columns are. Everything below is a consequence of just this.

**Consequence 0 — the posterior is a pushforward.** Using that independence,

$$P(S = s \mid E) = \sum_{d,t,p} P(S = s \mid d, t, p)\, P(d, t, p \mid E).$$

Read this as: take the outcome posterior $P(D, T, P \mid E)$ — a distribution over the 27 outcome cells — and push it through the fixed map $P(S \mid D, T, P)$. The scenario posterior is the *image* of the outcome posterior under that map; nothing else enters. In the limiting hard case $P(S = s \mid d, t, p) = \mathbb{1}[f(d,t,p) = s]$ for a labelling function $f$ that buckets the 27 cells into 3 groups, this collapses to

$$P(S = s \mid E) = \sum_{(d,t,p) \in f^{-1}(s)} P(d, t, p \mid E) = P\big((D, T, P) \in f^{-1}(s) \mid E\big).$$

The "scenario probability" is *literally* the posterior mass the outcome distribution puts on the region $f^{-1}(s)$. $S$ is a **coarsening** of $(D, T, P)$ — partition 27 cells into 3 buckets and sum. The soft columns (the actual Hormuz table) make it a *stochastic* coarsening rather than a hard partition, but the substance is unchanged.

**Consequence 1 — $S$ carries no information beyond the outcome posterior.** Because $S$ is a (stochastic) function of $(D, T, P)$, the outcome posterior $P(D, T, P \mid E)$ is a *sufficient statistic* for the scenario posterior: once you have it, you already know $P(S \mid E)$ exactly. $S$ has no degrees of freedom of its own — it is a derived view of the outcomes, not a new variable.

**Consequence 2 — you cannot set the scenario prior independently.** The marginal scenario prior is *induced*, not chosen:

$$P(S = s) = \sum_{d,t,p} P(S = s \mid d, t, p)\, P(d, t, p),$$

with $P(d, t, p)$ fixed by the upstream model. If a domain expert says *"a priori, Severe Closure should sit around 5% in this context,"* the labelling model has **no parameter to turn** — $P(\text{Severe})$ is whatever the outcome marginal and the partition imply. In the inverted model $P(S = \text{Severe} \mid M, C)$ is a primitive you write down, so "5% in this context" is simply a number entered.

**Consequence 3 — you cannot set per-regime likelihoods (Bayes factors) independently.** The governance quantity stakeholders ask for is the Bayes factor

$$\Lambda_{s_1, s_2}(E) = \frac{P(E \mid S = s_1)}{P(E \mid S = s_2)} \qquad \text{— "}E\text{ is }\Lambda\times\text{ more likely under }s_1\text{ than }s_2\text{."}$$

Everything hinges on $P(E \mid S = s)$; deriving it in each model shows the difference.

*Labelling model* ($D, T, P \to S$, $S$ a leaf). Starting from the definition and using $S \perp E \mid (D, T, P)$:

$$P(E \mid S = s) = \frac{P(E, S = s)}{P(S = s)} = \frac{\sum_{d,t,p} P(S = s \mid d, t, p)\, P(E, d, t, p)}{\sum_{d,t,p} P(S = s \mid d, t, p)\, P(d, t, p)}.$$

In the hard case, with $R_s = f^{-1}(s)$ the outcome region for scenario $s$, this is

$$P(E \mid S = s) = \frac{P\big(E,\, (D,T,P) \in R_s\big)}{P\big((D,T,P) \in R_s\big)} = P\big(E \mid (D,T,P) \in R_s\big).$$

This is **forced** — entirely determined by the existing upstream joint $P(E, D, T, P)$ and the partition $R_s$. There is no parameter to elicit; whatever the network already implies about how $E$ co-occurs with the region $R_s$ *is* the likelihood. And it is often **degenerate**: if $E$ is evidence on one of the very variables that define the regions, the ratio breaks. Suppose $R_{\text{Stress}}$ requires $D = \text{none}$ and the evidence is $E = \{D = \text{severe}\}$:

$$P(D = \text{severe} \mid S = \text{Stress}) = \frac{P\big(D = \text{severe},\, (D,T,P) \in R_{\text{Stress}}\big)}{P(R_{\text{Stress}})} = 0,$$

because $R_{\text{Stress}}$ forbids $D = \text{severe}$ — so $\Lambda_{\text{Severe}, \text{Stress}}(D = \text{severe}) = \infty$. "Infinitely more likely" is just the definition talking back, not an evidential statement.

*Latent regime model* ($S \to D, T, P$, $\text{Pa}(S) = \{M, C\}$). Now $P(E \mid S = s)$ is a genuine likelihood. For $E = \{D = d\}$:

$$P(D = d \mid S = s) = \sum_{m,c} P(D = d \mid s, m, c)\, P(m, c \mid s), \qquad P(m, c \mid s) = \frac{P(s \mid m, c)\, P(m, c)}{\sum_{m',c'} P(s \mid m', c')\, P(m', c')}.$$

The decisive difference: $P(D = d \mid s, m, c)$ is a **primitive emission CPT you elicit directly** — a free knob. The Bayes factor is then a clean, parent-averaged ratio of elicited likelihoods,

$$\Lambda_{s_1, s_2}(D = d) = \frac{\sum_{m,c} P(D = d \mid s_1, m, c)\, P(m, c \mid s_1)}{\sum_{m,c} P(D = d \mid s_2, m, c)\, P(m, c \mid s_2)},$$

and for soft evidence $\varepsilon$ (the translator's likelihood vector) it is the version derived in §A.2.6, $\Lambda_{s_1,s_2}(\text{article on } D) = \sum_d \varepsilon_d\, P(D{=}d \mid s_1, \ldots) \big/ \sum_d \varepsilon_d\, P(D{=}d \mid s_2, \ldots)$. No degeneracy: $P(D = \text{severe} \mid S = \text{Stress}, m, c)$ is a small-but-nonzero elicited number (a de-escalating regime *occasionally* still produces severe damage), so the ratio stays finite. And because emissions are conditionally independent given $S$, multiple articles **compose multiplicatively** — $\Lambda(E_1, E_2) = \Lambda(E_1)\,\Lambda(E_2)$ — which the labelling model cannot do, since its "evidence" all couples through the shared outcome regions.

*The same number, side by side.* Evidence $E = \{D = \text{severe}\}$, comparing Severe vs Stress:

| | Labelling model | Regime model |
|---|---|---|
| $P(E \mid \text{Severe})$ | $P(D{=}\text{severe} \mid (D,T,P) \in R_{\text{Severe}}) = 1$ (definitional) | $P(D{=}\text{severe} \mid \text{Severe}, m, c) \approx 0.70$ (elicited) |
| $P(E \mid \text{Stress})$ | $0$ (region forbids it) | $\approx 0.05$ (elicited) |
| $\Lambda_{\text{Severe},\text{Stress}}$ | $\infty$ — degenerate, restates the partition | $0.70 / 0.05 = 14$ — finite, interpretable |

**The crisp analogy.** Define a medical **syndrome** as "fever ∧ cough." Then $P(\text{syndrome})$ is fixed by $P(\text{fever}, \text{cough})$ — you cannot set its prevalence independently — and $P(\text{lab result} \mid \text{syndrome})$ is whatever the symptom region implies. Now instead model a latent **disease** that *causes* fever and cough: you get an independent disease prevalence *and* per-disease test likelihoods. The labelling model is the syndrome (symptoms → label); the inversion is the disease (cause → symptoms). Same data, different expressiveness.

**Expressiveness, not correctness.** The labelling model is not *wrong*. It coherently answers "what is the probability the outcomes fall in the Severe-Closure region, given the evidence" — a legitimate posterior. What it *cannot express* are the two things this product needs: (1) an independently-specifiable regime prior (Consequence 2), and (2) per-regime evidence likelihoods / clean Bayes factors (Consequence 3). The inversion adds exactly those two knobs. **This — not a "the posterior does not exist" framing — is the honest justification for reversing the arrows, and the one to put in front of a skeptical reviewer.**

**The cost.** The two new knobs are not free: they presuppose a semantic shift — scenarios must be read as *modal signatures*, not definitions — and the regime prior $P(S \mid M, C)$ trades off against the emission tails, so many (prior × emission) factorisations reproduce the same outcome marginals; this elicitation non-identifiability is the deliberate price of the inversion, developed in full (commitment, resolution, downstream consequences) in §A.2.4.

### A.2.4 Scenarios as modal signatures: the semantic commitment the inversion requires

§A.2.3 justified the inversion on expressiveness grounds and named its price: scenarios must be read as *modal signatures*, not *definitions*. That phrase is doing a lot of work, and it is precisely the point where a skeptical reviewer (Alex's objection — *"I can't follow the causal chain if 'conflict mitigates' is a latent variable instead of a prediction"*) is right to push. This section makes the commitment explicit, because it — not the algebra of §A.2.3 — is the thing the client must actually endorse.

**The tension, stated sharply.** There are two incompatible readings of what a "scenario" *is*:

- **Definitional.** A scenario *is* a region of outcome space: $\text{Severe Closure} \equiv (\text{severe} \wedge \text{long} \wedge \text{closed})$. The narrative is a definition; $(D, T, P)$ deterministically pin $S$, so $S = f(D, T, P)$. This is what the client's narratives literally say, and what the framework's own *outcome-sufficiency* test demands ("could you *uniquely* name the scenario from $\mathcal O$?"). It points to $\mathcal O \to S$ — the labelling model.
- **Generative.** A scenario is a latent regime that *causes/biases* outcomes, with stochastic, overlapping emissions. This is what the inversion $S \to \mathcal O$ needs in order to buy anything — if emissions don't overlap, the "generative model" is just a partition in disguise.

You cannot hold both crisply. The inversion silently assumes the generative reading while the narratives and the sufficiency test assert the definitional one. That unspoken switch is exactly what makes the causal chain unfollowable to a reviewer: as written, the two readings are conflated.

**The resolution.** Commit explicitly to the generative reading, and reinterpret each narrative as the **mode** of that regime's emission distribution — not a definition of it. Formally, the narrative for scenario $s$ is the claim

$$\big(d^\star_s,\, t^\star_s,\, p^\star_s\big) \;=\; \arg\max_{(d,t,p)} P\big(D = d, T = t, P = p \mid S = s,\ \text{context}\big),$$

e.g. $\arg\max P(D, T, P \mid S = \text{Severe}) = (\text{severe}, \text{long}, \text{closed})$. The signature is the **peak** of the emission distribution, and the distribution has nonzero mass **off** the peak: $P(D = \text{moderate} \mid S = \text{Severe}) > 0$. The narrative describes *where each regime's emissions concentrate*, not a wall around them. This single move resolves everything:

- It makes $S \to \mathcal O$ coherent. Emissions are genuine overlapping distributions, so $S$ is a real latent variable with an independent prior and clean Bayes factors — the expressiveness wins of §A.2.3 survive.
- It reinterprets *outcome sufficiency* correctly: from "outcomes deterministically identify the regime" to "the modal signatures are distinct enough that the regimes are statistically distinguishable from outcomes." The framework's parenthetical "(or near-uniquely)" stops being a hedge and becomes the actual definition.
- It dissolves the degeneracy. The $P(D = \text{severe} \mid \text{Stress}) = 0 \Rightarrow \Lambda = \infty$ pathology of §A.2.3 disappears, because off-mode mass is nonzero by construction.

**The semantic commitment the client must share.** Adopting modal signatures is not a mathematical fact one can prove — it is a modelling stance the client has to endorse. Concretely they must accept three things a definitional scenario does not require:

1. **Overlap.** A given real-world outcome (say, moderate damage) is consistent with several regimes at different probabilities. There is no longer a crisp "if you observe $X$, you are in scenario $Y$."
2. **Off-signature realisations.** "Severe Closure" can occasionally manifest with not-fully-severe outcomes, and a benign regime can occasionally throw a severe reading. The label names the *generating regime*, not a guaranteed outcome bundle.
3. **Permanent latency.** Because $S$ is only probabilistically tied to outcomes, you can *never* point at the world and certify "that was Severe Closure" — you only ever report a posterior. A definitional scenario is checkable in principle (did outcomes land in the region?); a regime is not.

**The licensing test.** There is a test for whether the commitment is even warranted: do these scenarios correspond to genuinely distinct underlying states of the world — an escalation dynamic, an Iranian decision posture — that drive many outcomes *together*? If yes, treating them as latent causes is natural and the inversion is right. If the scenarios are merely convenient descriptive buckets with no common underlying cause, then the generative story is causal lipstick on a taxonomy, and the honest model is the labelling one (reported plainly as a region-probability). The client/expert affirming *"these are real regimes, not just outcome buckets"* is the thing that licenses the inversion. **That affirmation belongs in the plan as an explicit precondition, not an assumption.**

**What it forces downstream (and why that's healthy).**

- **Elicitation (Plan 4).** The expert is now asked, per regime: *"If the world were truly in regime $s$ given context $(m, c)$, what is the distribution over damage / duration / path?"* — including the **off-mode tails**, brand-new judgments the labelling CPT never required. The modal cell is anchored to the narrative; the tails carry the regime's real informational content. Plan 1 §B already calls these "**anchor-derived** emission CPTs" — that construction *is* modal-signature thinking; it simply never names the commitment, which is why it currently reads as hand-wavy.
- **Validation.** A concrete QA check falls out: verify that the $\arg\max$ of each elicited emission distribution equals the client's stated narrative signature. Narrative-vs-CPT consistency becomes a *testable invariant*.
- **Dashboard communication (Plan 5).** The UI must not let stakeholders read "$P(\text{Severe Closure}) = 42\%$" as "42% chance outcomes will be severe / long / closed." It has to read as "42% posterior that the underlying *regime* is Severe Closure," ideally with the modal signature shown alongside ("Severe Closure typically looks like: severe / long / closed"). Otherwise stakeholders silently re-import the definitional reading and get confused when a "Severe Closure" forecast coexists with a non-severe damage forecast — the exact confusion the reviewer is voicing, now surfacing for the end user.

**Residual caveat.** Modal signatures fix the *coherence* but not the *identifiability* flagged in §A.2.3's "cost": narratives only pin the mode, leaving the prior $P(S \mid M, C)$ and the off-mode emission mass jointly under-determined. The semantic commitment is necessary but not sufficient; Plan 4 still needs an elicitation *order* (emissions anchored to modes first, prior from base rates second) to break the degeneracy.

**One-paragraph version (for the executive summary / client-facing doc).** Scenarios in this model are latent regimes, not outcome definitions. Each scenario's narrative specifies the modal (most-probable) outcome signature its regime generates, not a region that defines it: a regime concentrates probability on its signature but retains nonzero mass on off-signature outcomes, so outcomes identify the regime only probabilistically. This is a deliberate semantic commitment — it presumes the scenarios name real underlying states of the world that co-drive outcomes, and it must be one the client endorses. It is what licenses the $S \to \{D, T, P\}$ topology, and it is the reading the dashboard must reinforce, lest stakeholders silently revert to reading scenario probabilities as outcome-region probabilities.

### A.2.5 Why the reversal is necessary: what happens when the client observes an outcome

§A.2.3 made the structural case and §A.2.4 named its price; this section shows the inverted model *operating*, by walking a single concrete observation through both topologies. It is the cleanest demonstration of the difference, and the setup the soft-evidence case (§A.2.6) builds on. Take the concrete case *"the client observes $D = \text{severe}$ — energy infrastructure damage was severe."*

**Under the original $\mathcal O \to S$ topology** — $S$ is a leaf with parents $(D, T, P)$ — the only way to update beliefs about $S$ given $D$ alone is to marginalise the labelling CPT over what the network currently believes about $T$ and $P$:

$$P(S = s \mid D = \text{severe}) \;=\; \sum_{t, p} P(S = s \mid \text{severe}, t, p) \cdot P(T = t, P = p \mid D = \text{severe}).$$

This is the **expectation of the labelling function** under the conditional distribution of the other outcomes. Two operational problems follow:

1. **Context-dependent answer.** The same single observation $D = \text{severe}$ produces a different scenario "probability" depending on what the network currently believes about $T$ and $P$ — because those beliefs are the weights $P(T = t, P = p \mid D = \text{severe})$ in the sum above. There is no posterior on the regime, only an expected label that drifts as upstream beliefs about other outcomes shift.
2. **No Bayes factor.** The quantity stakeholders actually want for governance — *"observing severe damage is X× more likely under Severe_Closure than under Stress_Mitigates"* — is the Bayes factor $\Lambda_{s_1, s_2}(D = \text{severe}) = P(D = \text{severe} \mid S = s_1) / P(D = \text{severe} \mid S = s_2)$. In this topology $D$ is not a child of $S$, so $P(D \mid S)$ is not definable. The governance quantity has no home.

**Under the reversed $S \to \mathcal O$ topology** — $D$ is a child of $S$ with emission CPT $P(D \mid S, M, C)$ — observing $D = \text{severe}$ opens the collider at $D$ and propagates likelihood evidence *back* to $S$ via direct Bayes' rule. After marginalising the unobserved emissions $T, P$ (each of their CPTs sums to 1 over its own state) and the upstream chain except $(M, C)$:

$$P(S = s \mid D = \text{severe}) \;\propto\; \sum_{m, c} P(M = m, C = c) \cdot P(S = s \mid m, c) \cdot P(D = \text{severe} \mid s, m, c).$$

Here $P(M, C)$ is the joint marginal of $(M, C)$ produced by the upstream chain, $P(S \mid m, c)$ is the regime CPT, and $P(D \mid s, m, c)$ is the emission CPT — the standard Bayesian-network posterior factorisation for a latent variable with both upstream parents and downstream emissions. Three things now work:

1. **Genuine Bayesian posterior on the regime**, derived by Bayes' rule. The answer no longer depends on what the network "currently believes" about the *sibling outcomes* $T, P$ — those marginalise out cleanly because each unobserved-emission CPT is normalised. (It does still depend on beliefs about the *upstream context* $(M, C)$, through the prior $P(S \mid M, C)$ and the weights $P(m, c)$ — but that is correct: the regime posterior should be weighted by how likely each context is. Context-dependence is relocated upstream, not eliminated.) The observation $D = \text{severe}$ thus stands on its own *relative to the other outcomes* as evidence about $S$.
2. **Multiple emissions compose multiplicatively** by independence given $S$. Observing all of $D, T, P$ together gives
   $$P(S = s \mid d, t, p) \;\propto\; \sum_{m, c, \ldots} P(\text{Pa}) \cdot P(s \mid m, c) \cdot P(D = d \mid s, m, c) \cdot P(T = t \mid s, \ldots) \cdot P(P = p \mid s, \ldots),$$
   each observed emission contributing its own likelihood factor. This is the evidence accumulation the client brief calls for, and it composes by independence rather than by labelling-table lookup.
3. **Bayes factors are first-class outputs.** $\Lambda_{s_1, s_2}(D = \text{severe})$ reduces to a parent-averaged column ratio of the emission CPT — directly interpretable as "evidence strength against a regime hypothesis", exactly the governance quantity stakeholders want.

**The load-bearing point** is the one §A.2.3 makes structurally, now visible in the mechanics: under $\mathcal O \to S$ the single observation $D = \text{severe}$ can only move $S$ through context-dependent re-weighting of sibling outcomes — $S$ has no prior of its own (Consequence 2) and no per-state likelihood to elicit (Consequence 3), so the governance Bayes factor is either forced by the outcome partition or diverges when evidence touches a defining variable. Under $S \to \mathcal O$ the same observation enters as a genuine likelihood on a latent regime carrying its own prior. That — not any failure of the labelling model to produce *a* posterior — is what reversing the arrows buys.

### A.2.6 The soft-observation case: the translator-injected likelihood-ratio interface

§A.2.5 walked through a *hard* observation — the client states "$D = \text{severe}$" with certainty. In production the analyst rarely has that luxury: observations arrive as headlines that the translator (Plan 2) parses into *soft* evidence on emission nodes. Under the latent-regime topology the same Bayes-rule machinery handles this case, but only if the translator's output is shaped correctly. The fix is the M2/C5 finding addressed in Plan 2 §A1; this section restates it in the language of the latent-regime topology so the math is visible in one place.

**How pgmpy injects soft evidence — Pearl's virtual child.** pgmpy's `virtual_evidence` parameter implements a bookkeeping device from Pearl 1988: it bolts a fictional leaf node $V$ onto the emission node $N$ (no real-world referent), declares $V$ observed at some value $v$, and fills $V$'s CPT column $P(V = v \mid N = s_i)$ with the per-state vector you supply. That column is, by construction, a **likelihood** — a function of $s_i$ in $[0, \infty)$, not summing to 1. Conditioning on $V = v$ multiplies the likelihood into the joint via plain Bayes' rule:

$$P(N = s_i \mid V = v) \;\propto\; P(V = v \mid N = s_i) \cdot P(N = s_i).$$

You never see the phantom child — pgmpy adds, marks-observed, queries, and drops it silently — but the multiplication into the joint is real. **Whatever vector the translator produces is interpreted by pgmpy as a likelihood**, regardless of whether the translator was written with that contract in mind.

**The current (broken) interface — finding M2/C5.** The translator's system prompt asks for a sum-to-1 distribution over the node's states, which is *posterior-shaped*: it is the LLM's best estimate of $P(N = s_i \mid \text{article})$. Feeding a posterior $T_i = P(s_i \mid \text{article})$ to pgmpy as if it were a likelihood, then letting pgmpy multiply by the BN's prior $P(s_i)$, expands by Bayes' rule into:

$$P(N = s_i \mid V = v) \;\propto\; T_i \cdot P(s_i) \;=\; \frac{P(\text{article} \mid s_i) \cdot P(s_i)}{P(\text{article})} \cdot P(s_i) \;\propto\; P(\text{article} \mid s_i) \cdot P(s_i)^2.$$

The prior is squared. Every soft observation on an emission node currently enters the BN through a prior-squared interface. Plan 2 §A1 has the full numerical example on `Tanker_Incidents`; the bug is reproducible in three lines of pgmpy with an artificial network.

**The fix — Plan 2 A1's likelihood-ratio output.** The translator is re-prompted to emit max-pinned likelihood ratios

$$\varepsilon_i \;=\; \frac{P(\text{article} \mid N = s_i)}{\max_{i'} P(\text{article} \mid N = s_{i'})} \;\in\; (0, 1],$$

with the best-supported state pinned at $\varepsilon = 1$ and others fractions. Feeding $\varepsilon$ to pgmpy now gives a single multiplication by the prior, exactly as Bayes prescribes:

$$P(N = s_i \mid V = v) \;\propto\; \varepsilon_i \cdot P(s_i) \;\propto\; P(\text{article} \mid s_i) \cdot P(s_i) \;\propto\; P(s_i \mid \text{article}).$$

**Why the switch matters specifically under the latent-regime topology.** With $S$ generative and emissions $\{D, T, P\}$ as the regime's observable signals, soft evidence on any emission is the central evidence channel from the translator to the regime posterior. The hard-observation walkthrough in §A.2.5 already shows the Bayesian update at the emission node propagating through $P(D \mid S, M, C)$ back to $S$ via collider-opening — *the soft case is identical mechanics*, with the certain $\mathbb{1}[D = \text{severe}]$ replaced by the soft vector $\varepsilon$. Composing the emission CPT with the $\varepsilon$ vector and the regime prior gives the regime-level Bayes factor that decomposes a soft article into "how much does this evidence favour each scenario":

$$\Lambda_{s_1, s_2}(\text{article on } D) \;=\; \frac{\sum_d \varepsilon_d \cdot P(D = d \mid S = s_1, \ldots)}{\sum_d \varepsilon_d \cdot P(D = d \mid S = s_2, \ldots)},$$

with the emission likelihood appropriately parent-marginalised. Posterior-shaped translator output silently bakes the BN's emission prior into this ratio twice; likelihood-shaped output gives a clean Bayes factor that composes by multiplication across independent articles.

**The interfaces are co-designed.** Plan 1 needs likelihood-ratio inputs on emission nodes to deliver Bayes-factor outputs at the regime; Plan 2 A1 produces exactly those inputs. Until A1 ships, every translator-injected observation on $\{D, T, P\}$ enters the latent-regime machinery through a prior-squared interface, biasing the regime posterior by the same mechanism A1 fixes at the emission node. This is the substance behind §C's claim that *"Plan 2 A1's contract is already the right shape"* — that one-line summary collapses what this section has just unpacked.

### A.3 The reframe — what changes structurally

Three structural deltas to the current network suffice to bring it into framework compliance:

1. **Reverse the $S$-arrows.** Remove $\{D, T, P\} \to S$; add $S \to \{D, T, P\}$. This restores R4: each $O \in \mathcal O$ has $S$ as a parent.
2. **Add upstream parents to $S$.** Add $M \to S$ and $C \to S$. This makes $\text{Pa}(S) = \{M, C\}$ — see §A.4 for the choice. This satisfies R5 (with the documented $U_3$ blind spot).
3. **Replace the labelling CPT with emission CPTs and a regime CPT.** $P(S \mid D, T, P)$ is removed. New CPTs:
   - $P(D \mid S, M, C)$ — Damage emission.
   - $P(T \mid S, U_1, U_3, M)$ — Duration emission.
   - $P(P \mid S, U_1, U_3, U_2)$ — Diplomatic Path emission.
   - $P(S \mid M, C)$ — Regime CPT.

No other node gains or loses an edge. The upstream chain $(U_1, U_2, U_3, U_4) \to A \to K \to (M, C)$ is structurally untouched. $\mathcal D = \{O = $ Oil Price Regime $\}$ stays a child of $(C, D)$, also untouched (R6 holds).

### A.4 Choosing $\text{Pa}(S)$ for Hormuz

For Hormuz, $\text{Pa}(S) = \{M, C\}$ — the downstream-most layer of Hormuz mediators. The framework's R5a (d-connection) and R5b (parsimony) drive the choice; this section walks through the diagnostic that produces it.

**R5a — d-connection audit.** Applying framework §5 step 4 to Hormuz with $\text{Pa}(S) = \{M, C\}$:

| $V$ | Active chain to $\text{Pa}(S)$ in $\mathcal X \cup \mathcal M$ subgraph | Covered? |
|---|---|---|
| $U_1$ Negotiations | $U_1 \to K \to M$ | ✓ |
| $U_2$ Regime Stability | $U_2 \to A \to K \to M$ | ✓ |
| $U_3$ Mediation | *(no path — $U_3$'s only outgoing edges go to $T$ and $P$, both in $\mathcal O$)* | ✗ **blind spot** |
| $U_4$ Sanctions | $U_4 \to M$ direct; $U_4 \to A \to K \to M$ | ✓ |
| $A$ Militia | $A \to K \to M$ | ✓ |
| $K$ Tankers | $K \to M$ direct; $K \to C$ direct | ✓ |
| $M, C$ | in $\text{Pa}(S)$ | ✓ |

All upstream nodes except $U_3$ have an active chain path to $\{M, C\}$ within the $\mathcal X \cup \mathcal M$ subgraph. $U_3$'s only outgoing edges go to $T$ and $P$, both in $\mathcal O$ — making $U_3$ a documented blind spot (triage below).

**R5b — parsimony.** $\{M\}$ alone covers $\{U_1, U_2, U_4, A, K\}$. Adding $C$ is technically redundant for d-connection (every chain that reaches $C$ also reaches $M$). $C$ is retained because (i) it provides direct closure-evidence sensitivity to $S$ — observed strait closure is a strong scenario signal in the client's narratives — and (ii) the elicitation question *"given military response $m$ and closure status $c$, what's the regime prior?"* is more concrete than *"given military response $m$ alone…"*. The retention is documented as a deliberate parsimony exception in §D.

**$U_3$ blind-spot triage** (framework §5 step 6). Mediation news rarely arrives as standalone evidence in the headline corpus the translator processes — it typically arrives paired with diplomatic activity (which the translator routes to $P$) or duration-relevant developments (routed to $T$). Soft evidence on $P$ or $T$ opens the corresponding collider and propagates to $S$. In practice, mediation news *does* move $S$, transitively through outcome-channel routing rather than directly.

Three triage options exist per the framework:

| Option | Status |
|---|---|
| Accept and document | ✅ chosen for v1 |
| Extend $\text{Pa}(S)$ to $\{U_3, M, C\}$ | Deferred to Plan 4 Layer 2 elicitation, which decides whether expert opinion supports a defensible direct $U_3 \to S$ effect justifying the CPT growth from 9 to 18 columns |
| Add structural mediator | Not applicable — no natural process node sits between $U_3$ and $\{D, T, P\}$; mediation acts directly on diplomatic/duration outcomes |

### A.5 Topology, drawn

```
                          U₁   U₂   U₃   U₄          ← 𝒳 (unchanged priors)
                            \   |   |   /|
                             \  |   |  / |
                              \ ▼   ▼ ▼  ▼
                               A → ... → K          ← 𝓜 chain (unchanged)
                                       /|
                                      ▼ ▼
                                      M  C          ← 𝓜 downstream-most (Pa(S))
                                     │\ │\
                                     │ ▼▼ ▼
                                     │  S           ← S latent regime (intermediate)
                                     │ /│\
                                     ▼▼ ▼ ▼
                                     D  T  P        ← 𝓞 emissions
                                     ▲  ▲  ▲          (each has S + upstream causal parents)
                                     │  │  │
                                  (M,C) (U₁,U₃,M) (U₁,U₃,U₂)
                                     │
                                     ▼
                                     O              ← 𝓓 (unchanged: child of C and D)
```

The same three deltas of §A.3, decomposed to the atomic per-edge level for the implementation diff (§B.1):

1. Remove the three arrows $\{D, T, P\} \to S$.
2. Add three arrows $S \to \{D, T, P\}$.
3. Add the arrow $M \to S$.
4. Add the arrow $C \to S$.
5. Replace the labelling CPT $P(S \mid D, T, P)$ with the regime CPT $P(S \mid M, C)$ and the three emission CPTs that gain $S$ as an additional parent.

### A.6 What changes operationally — d-separation behaviour

Here is what evidence on each node type does to $P(S \mid E)$ under the latent-regime topology:

| Evidence on | Path to $S$ | Updates $P(S)$? |
|---|---|---|
| Any $U_i$ except $U_3$ | active chain $U_i \to \cdots \to M \to S$ (or $\to C \to S$) | ✓ |
| $U_3$ | only via $T$ or $P$ (colliders, blocked unless $T, P$ or $O$ observed) | ✗ blind spot, see §A.4 |
| $A, K$ | active chain to $M$ | ✓ |
| $M, C$ | direct parent of $S$ | ✓ |
| $D, T, P$ | direct child of $S$ | ✓ |
| $O$ | descendant of $D$ → opens collider at $D$ → activates upstream paths | ✓ |

Almost every upstream node has an active path to $S$ — propagation runs through R5's intended channel ($\text{Pa}(S) = \{M, C\}$) rather than through a labelling CPT. The militia, sanctions, military, and closure headlines that drive the scenario posterior continue to do so; the only operational caveat is the documented $U_3$-only blind spot.

### A.7 Relationship to the regime-switching HMM

A regime-switching HMM is in operation in a sibling inflation-forecasting workstream (separate codebase; the inflation HMM does not exist in this repository — see [docs/bn_hmm_integration.md](bn_hmm_integration.md) and master plan §6). The latent-regime BN is, structurally, **the static slice of exactly that style of HMM** with the transition channel replaced by the context-summary parents $\text{Pa}(S)$. Schematically:

```
Regime-switching HMM (temporal):

     S_{t-1} ──► S_t ──► S_{t+1}        ← transition matrix P(S_t | S_{t-1})
                 │
                 ▼
                 X_t                     ← emission P(X_t | S_t)


Latent-regime BN (static, Hormuz instance):

      Pa(S) = {M, C} ──► S               ← regime CPT P(S | M, C)
                         │                  (the "transition channel" specialised
                         ▼                   to a static context-summary)
                         O_i              ← emissions P(O_i | S, upstream)
```

The static slice maps exactly: $\text{Pa}(S)$ in the BN plays the role that $S_{t-1}$ plays in the HMM (a context-driven prior over the current regime), and $\{D, T, P\}$ play the role of $X_t$. The inference structure the team is already comfortable with for the HMM — *treat $S$ as latent, model the data as emissions, infer the regime by Bayes' rule* — carries over to the BN directly. The client's "predict three scenario probabilities" requirement is honoured by the same kind of object on both sides (a real posterior over a latent regime), where before the BN was reporting the expectation of a labelling function dressed as a probability.

Glide-path consequence: a future workstream layering temporal dynamics onto the BN (a true transition matrix $P(S_t \mid S_{t-1})$ in addition to the context-summary parents, or coupling to the existing inflation HMM per `docs/bn_hmm_integration.md`) has a well-defined docking point at $S$. The temporal extension itself remains out of scope for this plan and for Plans 2–5.

### A.8 Math summary

- **BN factorisation and inference.** A Bayesian network expresses the joint $P(X_1, \ldots, X_n) = \prod_i P(X_i \mid \text{Pa}(X_i))$. Arrows encode the *factorisation*, not data flow; inference is direction-agnostic. The reframe changes which CPTs appear in the product — the labelling CPT $P(S \mid D, T, P)$ is replaced by the regime CPT $P(S \mid M, C)$ plus three emission CPTs with $S$ as an additional parent — but the factorisation framework is identical.
- **Entropy diagnostic for M1.** Shannon entropy $H(p) = -\sum_i p_i \log_2 p_i$ on a 3-state distribution ranges in $[0, \log_2 3] \approx [0, 1.585]$ bits. Computed over all 27 columns of the current $P(S \mid D, T, P)$, only the two *extreme* corners are sharp ($H = 0.37$ at `none/short/open`, $H = 0.52$ at `severe/long/closed`); the rest is diffuse (20 of 27 columns at $H \ge 1.1$, mean $1.16$, max $1.54$ at `severe/short/open`), and the middle regime is never sharply identified. Two razor-sharp peaks over a broad high-entropy mass — a *continuum*, not a clean two-cluster split — is the diffuse-classifier signature of the labelling-CPT trap (framework §10). Full diagnostic in §A.2.
- **Regime posterior factorisation.** With $\text{Pa}(S) = \{M, C\}$:

  $$P(S = s \mid E) \;\propto\; \sum_{m, c} P(M = m, C = c \mid E_{\text{up}}) \cdot P(S = s \mid m, c) \cdot P(E_{\text{down}} \mid S = s, m, c, \ldots).$$

  The "$\propto$" absorbs a factor of $P(E_{\text{up}})$ that arises if one writes the chain rule with the joint $P(M, C, E_{\text{up}})$ instead of the conditional $P(M, C \mid E_{\text{up}})$ used here; it is constant in $s$ and therefore drops out of the normalisation over scenario states. Exact inference (variable elimination) in pgmpy computes this sum directly given the topology.

- **Bayes factors as first-class outputs.** $\Lambda_{s_1, s_2} = P(E \mid S = s_1) / P(E \mid S = s_2)$ — the evidence strength against a regime hypothesis. Composes by independence: $\Lambda(E_1, E_2) = \Lambda(E_1) \cdot \Lambda(E_2)$ if $E_1, E_2$ are conditionally independent given $S$. This is the quantity stakeholders actually want for governance (*"evidence X is 8.7× more likely under Severe than under Stress"*) and the current model cannot produce it. Extraction mechanics in §B.1 item 3.
- **D-separation rules.** Chain $A \to B \to C$ and fork $A \leftarrow B \to C$: $B$ blocks when conditioned on, otherwise active. Collider $A \to B \leftarrow C$: opposite — blocks by default, opens when $B$ or any descendant is conditioned on. These are the rules underwriting the §A.6 d-separation table.
- **Dirichlet-based parameter uncertainty.** Existing resampling $\theta^{(m)} \sim \text{Dirichlet}(\kappa \cdot \theta_{\text{elicited}})$ transfers unchanged in form. The labelling CPT $P(S \mid D, T, P)$ leaves the resampled set; the three emission CPTs (each with $S$ as one extra parent) and the new regime CPT $P(S \mid M, C)$ join it. Per-category $\kappa$ defaults follow framework §9: emission CPTs and the regime CPT at $\kappa = 10$ (regime-conditional generation, genuinely uncertain), upstream-chain CPTs at $\kappa \in [20, 50]$ (historical pattern-matching, moderate confidence). Plan 1 ships at $\kappa = 10$ on the new CPTs pending elicited replacements from Plan 4 Layer 4. Note: pgmpy direct VE consumes the point-estimate CPT only, so Plan 1 (pgmpy-only by design) is insensitive to the $\kappa$ choice; $\kappa$ first becomes operative when `PymcBackend` lands in Plan 3 Phase 2 (hierarchical priors).

---

## Section B — Engineering implementation

### B.1 Scope

Three pieces of work, with **no engineering prerequisites**. Plan 1 ships before Plan 2 begins, editing the current code directly — `src/network.py`, `src/cpt_data.py`, and `src/inference.py`. No declarative `NetworkSpec`, no backend abstraction, no `PymcBackend`: pgmpy already supports variable elimination on the latent-regime topology via the existing `BNInferenceEngine`, and the three-clamped-inferences pattern for Bayes-factor extraction is a small helper alongside it. Plan 3 later lifts this work into the declarative `NetworkSpec` / dual-backend architecture (Phase 0 → Phase 2), at which point PyMC-native support for the latent regime is added.

1. **Topology rewire in `src/network.py`.** Remove the edges $\{D, T, P\} \to S$ and add $S \to \{D, T, P\}$, $M \to S$, $C \to S$. Drop the labelling CPT $P(S \mid D, T, P)$ from the network construction; wire in the four new CPTs (item 2). No flag toggle — the latent-regime topology is the only topology after Plan 1 ships. The legacy labelling code path is removed in the same change.
2. **Emission CPTs and regime CPT.** The labelling CPT $P(S \mid D, T, P)$ is removed. New CPTs:
   - $P(D \mid S, M, C)$ — Damage emission.
   - $P(T \mid S, U_1, U_3, M)$ — Duration emission.
   - $P(P \mid S, U_1, U_3, U_2)$ — Diplomatic Path emission.
   - $P(S \mid M, C)$ — Regime CPT.

   These CPTs are **anchor-derived in Plan 1 and re-elicited in Plan 4 Layer 2**. Plan 1 ships with one-off anchor values constructed deterministically from the current model. The procedure is:

   1. Run the current BN (`build_network()`) with no evidence to obtain the implied joint $P_{\text{cur}}(D, T, P, M, C, U_1, U_2, U_3, \ldots)$ over the relevant parent sets.
   2. Combine with the current labelling CPT to obtain the augmented joint
      $\tilde P(S, D, T, P, M, C, U_1, U_2, U_3, \ldots) = P(S \mid D, T, P) \cdot P_{\text{cur}}(D, T, P, M, C, U_1, U_2, U_3, \ldots)$.
   3. Marginalise $\tilde P$ to each emission CPT's parent set and divide:
      - $P(D \mid S, M, C) = \tilde P(D, S, M, C) / \tilde P(S, M, C)$.
      - $P(T \mid S, U_1, U_3, M) = \tilde P(T, S, U_1, U_3, M) / \tilde P(S, U_1, U_3, M)$.
      - $P(P \mid S, U_1, U_3, U_2) = \tilde P(P, S, U_1, U_3, U_2) / \tilde P(S, U_1, U_3, U_2)$.
   4. **Regime CPT.** $P(S \mid M, C) = \tilde P(S, M, C) / \tilde P(M, C)$.

   This procedure is reproducible from the current `src/network.py`, and it is a principled bootstrap — but its two halves carry very different epistemic weight, and §A.2.4 is the reason. The **modes** transfer faithfully: $\arg\max_d P(D \mid S{=}s, M, C)$ inherits the dominant outcome region the old classifier already associated with each scenario, so the emission peaks should line up with the client's narrative signatures (checked, not assumed — see §B.3). The **off-mode tails**, by contrast, are an *artifact of the inversion*, not a belief about regime behaviour: the old labelling CPT was never elicited as a set of regime emissions, so $P(D{=}\text{moderate} \mid S{=}\text{Severe}, M, C)$ falls out as whatever mass the upstream joint happens to leave in cells the classifier still attributes partly to Severe — i.e. *classifier-boundary fuzziness*, not "how often a Severe regime produces moderate damage." Since §A.2.4 makes precisely those tails the regime's real informational content (they are what dissolve the $\Lambda = \infty$ degeneracy and carry each regime's distinguishing signal), the honest statement is: **the bootstrap pins the modes and supplies degeneracy-safe non-zero tails, but the tail *magnitudes* carry no genuine regime information until Plan 4 Layer 2 re-elicits them.** This is an acceptable v1 — it ships a coherent, non-degenerate model whose peaks match the narratives and whose Bayes factors are finite — *provided the plan says exactly this* rather than overselling the tails as elicited belief. The procedure is implemented as a one-off offline script at `scripts/derive_latent_regime_anchors.py`; its outputs are committed as literal CPT values in `src/cpt_data.py` and consumed at inference time without re-deriving. The script itself is committed alongside its outputs to provide reproducibility and an audit trail for how the v1 CPTs were derived.

   **Concentrations.** Provisional $\kappa = 10$ on every emission CPT and on $P(S \mid M, C)$ — the framework §9 default for regime-conditional CPTs whose epistemic basis is genuinely uncertain rather than empirical historical pattern-matching. Plan 4 Layer 4 round-trips elicited per-CPT $\kappa$ values into `cpt_provenance` and `PymcBackend` consumes them from there. Plan 1 is therefore correct-but-provisional on the new CPT parameters; Plan 4 produces the defensible elicited replacements.

3. **Bayes-factor extraction helper in `src/inference.py`.** Add a function alongside `BNInferenceEngine` that implements the **three-clamped-inferences pattern** (framework §8.1): pre-compute $P_{\text{marg}}(S = s)$ via one VE call with no evidence; for each $s$, clamp $S = s$ and run VE with evidence $E$ to obtain $P(E, S = s)$; divide to get $P(E \mid S = s)$ and form Bayes factors $\Lambda_{s_1, s_2} = P(E \mid S = s_1) / P(E \mid S = s_2)$. Clamping $S = s$ leaves $\text{Pa}(S)$ in the model — pgmpy marginalises them as part of standard VE. No abstraction layer is introduced; the helper calls into `BNInferenceEngine` three times. This implementation is the canonical reference for the latent regime — Plan 3 Phase 1 wraps it inside `PgmpyBackend`, and Plan 3 Phase 2 adds PyMC-native latent-regime support (a `pm.Categorical("S", p=cpt_table[m_idx, c_idx])` indexed by the current $(M, C)$ values, with analytic marginalisation of $S$ at $|S| = 3$ per framework §8.2) validated for parity against the pgmpy reference per §B.3.

### B.2 Deliverables

- **`docs/scenario_bn_framework.md`** — the foundational write-up of the scenario-as-latent BN framework. Reusable IP, lives outside the numbered plan series. Can proceed in parallel with engineering.
- **`src/network.py`** — rewired DAG (edges $S \to D, T, P$, $M \to S$, $C \to S$ added; legacy $\{D, T, P\} \to S$ removed). The latent-regime topology is the only topology after Plan 1 ships; the legacy labelling code path is deleted in the same change.
- **`src/cpt_data.py`** — anchor-derived emission CPTs ($P(D \mid S, M, C)$, $P(T \mid S, U_1, U_3, M)$, $P(P \mid S, U_1, U_3, U_2)$) and the regime CPT $P(S \mid M, C)$, with provisional per-CPT $\kappa = 10$ values (replaced by elicited values in Plan 4 Layer 4 once `PymcBackend` consumes them).
- **`src/inference.py`** — Bayes-factor extraction helper (three-clamped-inferences pattern) sitting alongside the existing `BNInferenceEngine`. No backend abstraction.
- **`scripts/derive_latent_regime_anchors.py`** — one-off offline script implementing the deterministic inversion procedure in §B.1 item 2. Outputs are committed as literal values in `src/cpt_data.py`; the script is committed alongside for reproducibility and audit.

Subsequent Plan 3 phases lift these deliverables into the dual-backend architecture: Phase 0 extracts `src/network.py` into a declarative `NetworkSpec`; Phase 1 wraps `BNInferenceEngine` and the Bayes-factor helper inside `PgmpyBackend`; Phase 2 adds `PymcBackend` (handling the latent-regime topology natively per §B.1 item 3) and validates parity against Plan 1's pgmpy implementation.

### B.3 Validation

- **D-separation positive check (R5a verification).** For each $V \in \mathcal X \cup \mathcal M$ that satisfies R5a per §A.4 (i.e., not $U_3$), apply a single hard evidence on $V$ and confirm $P(S \mid V) \neq \pi_{\text{marg}}(S)$ to floating-point tolerance. This is the framework's positive d-connection check.

- **D-separation blind-spot check (documented R5a failure).** Apply hard evidence on $U_3$ alone and confirm $P(S \mid U_3) = \pi_{\text{marg}}(S)$ to floating-point tolerance. This is the explicit documentation that the mediation blind spot is intentional, not an inference bug. **This check turning *negative* is acceptable** — but the result is recorded so future maintainers see the design choice as a deliberate trade-off, not a regression.

- **Qualitative posteriors.** The latent-regime model produces qualitatively-sensible posteriors on canonical test evidence: escalation sequence (sanctions tightening → high militia → major military → full closure) pushes Severe up; de-escalation sequence (negotiations success → low militia → none military → no closure) pushes Stress up.

- **Bayes factors.** $\Lambda_{s_1, s_2}$ computable on the canonical evidence configurations and consistent with the regime posterior shift implied by Bayes' rule.

- **Anchor-CPT mode check (the §A.2.4 narrative invariant, applied to the bootstrap).** For each emission CPT produced by `derive_latent_regime_anchors.py`, confirm that $\arg\max$ over the emitted variable, at that scenario's modal parent context, equals the client's stated narrative signature — e.g. $\arg\max_d P(D \mid S{=}\text{Severe}, M{=}\text{major}, C{=}\text{closed}) = \text{severe}$. This applies §A.2.4's validation invariant to *Plan 1's bootstrap*, not only to Plan 4's elicitation: if the inverted labelling CPT misplaces a mode, that is the signal the old classifier was too soft to bootstrap from, and the offending CPT is hand-corrected (or the narrative re-examined) before ship. Reported per (scenario, context) cell, so a miss is localised rather than hidden in an aggregate.

- **Non-degeneracy floor (preserve finite Bayes factors).** Confirm every emission-CPT cell is strictly positive — no exact zeros — so the $\Lambda = \infty$ degeneracy that §A.2.3 / §A.2.4 dissolve cannot re-enter through a bootstrapped zero tail. The inversion already yields non-zero tails wherever the labelling CPT assigns non-zero off-mode mass (verified: e.g. the `severe/long/closed` column is $[0.01, 0.09, 0.90]$ — no zeros), but the check enforces a small floor $\varepsilon_{\min}$ and renormalises if any derived cell rounds to zero, guaranteeing finite, composable Bayes factors from the gate. This is the engineering guarantee behind §A.2.4's claim that off-mode mass is "non-zero by construction."

- **Synthetic-data calibration.** Sample from the new model under a known true regime $s^\star$, run inference on the simulated evidence, check that the average log-Bayes-factor $\log \Lambda_{s^\star, s}$ in favour of the true regime grows roughly linearly in the number of independent emissions observed. Standard self-consistency check for any latent-variable inference; protects against silent emission-CPT errors.

- **M7 closure check — point-estimate vs resample-mean gap.** Master-plan §4 maps M7 to this plan on the theoretical argument in §A.2 (the labelling-CPT non-linearity drives the original 1–3pp gap; with labelling removed, only the residual Jensen gap survives). The check operationalises that argument: on a battery of $\geq 20$ canonical evidence configurations spanning the simplex (no-evidence, single-piece-of-evidence per node, escalation sequence, de-escalation sequence, contradictory evidence), compute $P(S \mid E)$ via (a) point-estimate VE on the elicited CPTs and (b) the existing resample-mean path (Dirichlet draws around each CPT). Report the per-scenario gap distribution; gate M7 as closed when the 95th-percentile gap is below the dashboard's display precision (0.5pp at current rendering). If any configuration exceeds the threshold, the diagnostic output identifies which emission CPT is driving the residual non-linearity, and the entropy diagnostic from §A.2 is re-run on that CPT to check for residual labelling-rule structure.

- **Backend parity (deferred to Plan 3 Phase 2).** Once `PymcBackend` lands, it must produce regime posteriors agreeing with `PgmpyBackend` (via direct VE or the three-clamped-inferences pattern) within MCMC error on a battery of canonical test evidence configurations. Until Phase 2 lands, the pgmpy implementation is its own reference and the remaining validation checks above are sufficient.

### B.4 Dependencies and sequencing

- **Plan 1 has no engineering prerequisites.** It edits the current `src/network.py`, `src/cpt_data.py`, and `src/inference.py` directly. The pgmpy `BNInferenceEngine` already supports variable elimination on the latent-regime topology; the three-clamped-inferences pattern is a helper alongside it, requiring no abstraction layer. Plan 1 ships before Plan 2 begins.
- **Plan 3 lifts Plan 1's work into the declarative architecture.** Phase 0 extracts the modified `src/network.py` into a `NetworkSpec`; Phase 1 wraps `BNInferenceEngine` and the Bayes-factor helper inside `PgmpyBackend`; Phase 2 adds `PymcBackend` with PyMC-native latent-regime support (per §B.1 item 3), validated for parity against Plan 1's pgmpy implementation.
- **Plan 3 Phases 3–4 build on the regime topology.** Phase 3's continuous-variable mechanism layers on the latent-regime topology — direct dependency for the regime's own emissions $\{D, T, P\}$ (any of which could later go continuous and would then be continuous emissions of $S$) and transitive for Oil_Price (Phase 4's first production migration, which sits in $\mathcal D$ rather than $\mathcal O$ in Hormuz v1 — see §E open question on $S \to O$ — through the new $P(D \mid S, M, C)$ along the $S \to D \to O$ chain).

```
Plan 1 (this plan, Section B — edits to src/network.py, src/cpt_data.py, src/inference.py on the existing pgmpy code path)
  → Plan 2 (translator robustification)
    → Plan 3 Phase 0 (NetworkSpec — lifts the post-Plan-1 src/network.py)
      → Plan 3 Phase 1 (PgmpyBackend — wraps BNInferenceEngine + Plan 1's Bayes-factor helper)
        → Plan 3 Phase 2 (PymcBackend, discrete; adds PyMC-native latent regime; parity vs Plan 1's pgmpy)
          → Plan 3 Phase 3 (continuous variables)
            → Plan 3 Phase 4 (Oil_Price migration)
              → Plan 4 (elicitation)
                → Plan 5 (dashboard)
```

**The framework write-up [docs/scenario_bn_framework.md](scenario_bn_framework.md) has no engineering prerequisites.** It is a documentation deliverable and can be drafted at any time. Drafting before engineering starts is recommended so that the engineering work and the cross-cutting touchpoints in §C can reference the framework consistently.

---

## Section C — Cross-cutting impact on other plans

### Plan 2 — Translator (`02_translator_robustification.md`)

Plan 2 A1 (likelihood semantics) commits the translator to producing relative evidence weights $\varepsilon_s = P(\text{article} \mid \text{state} = s) / \max_{s'} P(\text{article} \mid \text{state} = s')$ — i.e., likelihood ratios rather than posterior-shaped distributions. Under the latent-regime topology, these per-evidence-channel likelihoods over the emission node's states are exactly what feeds the regime-level Bayes-factor decomposition $\Lambda_{s_1, s_2}$: the emission CPT propagates the $\varepsilon_s$ values into regime-level Bayes factors. **No additional translator work is required** — Plan 2 A1's contract is already the right shape.

Plan 2 also carries an **optional pairwise-Bayes-factor elicitation variant** (A1, deferred) that asks the LLM directly for emission-node ratios $\Lambda^{\text{emis}}_{ij} = P(\text{article} \mid N{=}s_i)/P(\text{article} \mid N{=}s_j)$ rather than absolute per-state likelihoods, so the model's implicit prior cancels in the ratio. This is the same Bayes-factor object this plan works in, one level down (emission states rather than regime states), and composes through the emission CPT into the regime-level $\Lambda$ identically — so if it ships, it is a *cleaner* feed to the latent regime, not a different contract. Its emission-level $\varepsilon$ vector is recovered by max-pinning, so Section B's engineering is unchanged either way.

One implementation hint: translator routing should target emission nodes ($\mathcal O$) and mediator nodes ($\mathcal M$) in roughly equal measure. Headlines about *outcomes* (damage, duration, diplomatic developments) route to $\mathcal O$; headlines about *process* (militia activity, tanker incidents, military posture) route to $\mathcal M$. Both channels propagate to $S$ — the latter via the chain through $\text{Pa}(S)$, the former by opening the collider at the emission node.

### Plan 3 — Inference engine (`03_pymc_integration_plan.md`)

Plan 3 runs after Plan 1 (and Plan 2) and lifts Plan 1's directly-edited files into the declarative `NetworkSpec` / dual-backend architecture:

- **Phase 0** extracts the modified `src/network.py` (carrying Plan 1's latent-regime topology) into a `NetworkSpec`.
- **Phase 1** wraps the existing inference machinery (`BNInferenceEngine` plus Plan 1's Bayes-factor helper) inside `PgmpyBackend`.
- **Phase 2** adds `PymcBackend` for discrete networks, including PyMC-native latent-regime support per §B.1 item 3, validated for parity against Plan 1's pgmpy implementation.
- **Phases 3–4** layer continuous variables (Phase 3) and the Oil_Price migration (Phase 4) on top of the regime topology — direct dependency for $\{D, T, P\}$ if they go continuous, transitive for Oil_Price (which sits in $\mathcal D$, not $\mathcal O$) via the new $P(D \mid S, M, C)$.

Plan 1 has no dependency on Plan 3. The pgmpy `BNInferenceEngine` already supports VE on the latent-regime topology; the three-clamped-inferences pattern is a small helper alongside it, requiring no abstraction layer.

### Plan 4 — Elicitation methodology (`04_elicitation_tool_plan.md`)

Plan 4 Layer 2 elicits the four new CPTs — three emissions $P(D \mid S, \ldots)$, $P(T \mid S, \ldots)$, $P(P \mid S, \ldots)$ and the regime CPT $P(S \mid M, C)$ — that replace the deleted labelling CPT $P(S \mid D, T, P)$. *Why* the questions invert (from labelling "given outcomes, which regime?" to generative-and-context-conditional "given the regime and context, what do outcomes look like?"), why the off-mode tails carry the regime's real content, and why emissions must be anchored before the prior to break the identifiability degeneracy — all of that is the canonical treatment in §A.2.4 and is not restated here. What is genuinely Plan 4's concern:

- **Protocol mapping.** The framework's §6 elicitation recipe is the elicitor's playbook; Layer 2 maps it onto a concrete expert-elicitation protocol (Cooke's classical model / SHELF / IDEA) for these four CPTs, and supplies the §A.2.4 validation invariant ($\arg\max$ of each elicited emission must equal the client's stated narrative signature) as an acceptance check.
- **κ routing.** Layer 4 round-trips elicited per-CPT $\kappa$ values into `cpt_provenance`, which `PymcBackend` consumes via the provenance pathway — replacing Plan 1's provisional uniform $\kappa = 10$.
- **Open question — does $U_3$ join $\text{Pa}(S)$?** Layer 2 decides whether expert opinion supports a defensible direct $U_3 \to S$ effect justifying $\text{Pa}(S) = \{U_3, M, C\}$ ($P(S \mid U_3, M, C)$, 18 columns instead of 9). The §A.4 blind-spot triage leaves this open; if no, Plan 1's $\{M, C\}$ ships unchanged.

### Plan 5 — Dashboard UI (`05_dashboard_ui_plan.md`)

Bayes factors as first-class outputs change how the dashboard communicates evidence strength.

- **Plan 5 C4 (rich observed-node panel)** exposes the per-observation Bayes-factor contribution once Plan 1's engineering lands; before that, C4 falls back to a percentage-point delta display.
- **Plan 5 C7 (before/after delta on new observation)** remains a percentage-point chip but gains a natural extension to a Bayes-factor mode under the latent-regime branch.
- **Plan 5 C6 (parameter vs forecast uncertainty distinction)** is unaffected by the reframe itself but benefits from the cleaner regime-posterior outputs.

**Stakeholder narrative.** Almost all upstream evidence shifts the regime posterior via the $\text{Pa}(S)$ channel. The only stakeholder caveat is the $U_3$-only blind spot (rare in practice; see §A.4). Plan 5 C4/C7 copy can describe the model as "evidence updates the regime posterior" without further qualification.

---

## Section D — Design decisions resolved

1. **Framework adoption.** Plan 1 instantiates the scenario-as-latent BN framework specified in [docs/scenario_bn_framework.md](scenario_bn_framework.md). Five node categories ($\mathcal X, \mathcal M, \mathcal O, S, \mathcal D$) and six edge rules (R1–R6). All sections in this plan are framed in framework language.
2. **Topology.** $S$ is an internal latent. $S \to D, T, P$ replaces $D, T, P \to S$ (R4 restoration). $\text{Pa}(S) = \{M, C\}$ (R5 satisfaction modulo the $U_3$ blind spot).
3. **Parent set $\text{Pa}(S) = \{M, C\}$ for v1.** Satisfies R5a for $\{U_1, U_2, U_4, A, K\}$. $U_3$ documented as an accepted blind spot (see §A.4 and §E). $\{M\}$-only is rejected on R5b parsimony-exception grounds (closure-evidence sensitivity narrative; see §A.4).
4. **Backend approach: Plan 1 ships on the existing pgmpy code path; Plan 3 lifts it into the dual-backend architecture later.** Plan 1 edits `src/network.py`, `src/cpt_data.py`, and `src/inference.py` directly — no `NetworkSpec`, no `PgmpyBackend`, no `PymcBackend`. The three-clamped-inferences pattern for Bayes-factor extraction is a helper alongside the existing `BNInferenceEngine`. Plan 3 Phase 0 lifts the modified `src/network.py` into a `NetworkSpec`; Phase 1 wraps `BNInferenceEngine` and the Bayes-factor helper inside `PgmpyBackend`; Phase 2 adds `PymcBackend` with PyMC-native latent-regime support, validated for parity against Plan 1's pgmpy implementation. After Phase 2, the dispatcher routes to either backend for latent-regime specs.
5. **Anchor-derivation procedure.** B.1 item 2's deterministic inversion (run current BN, multiply by current labelling CPT, marginalise + divide) produces all four new CPTs as a v1 bootstrap. Honest scope (per §B.1 and §A.2.4): the inversion transfers the emission **modes** faithfully (peaks match the narratives, checked in §B.3) and yields degeneracy-safe non-zero **tails**, but the tail magnitudes are classifier-boundary fuzziness, not regime-emission belief, and carry no genuine regime information until Plan 4 Layer 2 re-elicits them.
6. **Per-CPT $\kappa$ values: provisional in Plan 1, elicited in Plan 4 Layer 4.** Plan 1 ships uniform $\kappa = 10$ on the emission CPTs and $P(S \mid M, C)$, matching framework §9 for regime-conditional CPTs. Plan 4 Layer 4 round-trips elicited per-CPT $\kappa$ values into `cpt_provenance`.
7. **Framework write-up as a Plan 1 deliverable.** [docs/scenario_bn_framework.md](scenario_bn_framework.md) is owned by Plan 1 rather than spun out as a separate doc plan. The framework is most defensible when paired with its first worked instance. The write-up lives in its own companion file (this plan's `01_latent_regime_plan.md` references it but does not contain it).
8. **Plan 1 ownership: conceptual + engineering tracked here; framework write-up in the companion file.** The conceptual decision and the engineering implementation are tracked in this plan; the framework write-up lives in the companion file `docs/scenario_bn_framework.md` per decision 7.

---

## Section E — Open questions

| Question | Block | Notes |
| --- | --- | --- |
| Sampler choice for the latent regime in `PymcBackend` | Plan 3 Phase 2 | Default: analytic marginalisation for $S$ (3-state, low cardinality, exact). NUTS + CompoundStep for larger discrete latents. Decide if/when the regime is generalised beyond 3 states. (Out of scope for Plan 1, which is pgmpy-only.) |
| Should $U_3$ join $\text{Pa}(S)$? | Section A.4 / Plan 4 Layer 2 | v1 ships with $\{M, C\}$. The mediation blind spot is operationally rare (translator typically routes mediation news to $P$ alongside any $U_3$ update). Plan 4 Layer 2 decides whether expert opinion supports a defensible direct $U_3 \to S$ effect justifying CPT growth from 9 to 18 columns. |
| Per-CPT $\kappa$ values | Section B engineering (resolved: provisional in Plan 1, elicited in Plan 4 Layer 4) | Plan 1 ships uniform $\kappa = 10$ on the emission CPTs and $P(S \mid M, C)$, matching framework §9. Plan 4 Layer 4 round-trips elicited values into `cpt_provenance`. |
| Should $S$ directly modulate $O$ (oil price)? | Section A.3 / future v2 | Framework §11 lists $S \to \mathcal D$ as a defensible extension. Hormuz v1 leaves it out: $O$ inherits regime sensitivity transitively via $D$ (and $C$), and a fresh elicitation of $P(O \mid C, D, S)$ would be needed. Reopen if expert opinion supports a regime-specific oil-price effect beyond what the existing chain captures. |

---

## Section F — Execution order summary

| Order | Item | Resolves | Rationale |
| --- | --- | --- | --- |
| Done | Conceptual decision (latent regime; $\text{Pa}(S) = \{M, C\}$) | M1 framing | Decision settled; recorded in §A and §D. |
| 1 (parallel) | Framework write-up [docs/scenario_bn_framework.md](scenario_bn_framework.md) | Reusable IP | No engineering prerequisites; can ship at any time. |
| 2 | Engineering implementation (Section B) | M1 implementation, M7 dissolution | No engineering prerequisites; edits `src/network.py`, `src/cpt_data.py`, and `src/inference.py` directly on the existing pgmpy code path. Ships before Plan 2 begins. Delivers the latent regime in production; produces Bayes factors as first-class outputs. Plan 3 later lifts this work into `NetworkSpec` / `PgmpyBackend` (Phases 0–1) and adds PyMC-native support (Phase 2). |

---

**End of plan.** Foundational reference: [docs/scenario_bn_framework.md](scenario_bn_framework.md). Companion plans: [docs/02_translator_robustification.md](02_translator_robustification.md), [docs/03_pymc_integration_plan.md](03_pymc_integration_plan.md), [docs/04_elicitation_tool_plan.md](04_elicitation_tool_plan.md), [docs/05_dashboard_ui_plan.md](05_dashboard_ui_plan.md). Orchestrator: [docs/master_plan.md](master_plan.md).
