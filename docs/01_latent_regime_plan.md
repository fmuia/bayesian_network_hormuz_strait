# Plan 1 — Latent Regime Reframing (Hormuz instance)

> **Status.** Conceptual decision ✅ resolved. Framework write-up ⬜ not started. Engineering implementation ⬜ not started.
>
> **Position in the sequencing.** First plan in the programme. The **conceptual decision** is settled and underlies the specification of Plans 2–5. The **framework write-up** (see B.2) and the **engineering implementation** are the outstanding work. Plan 1's engineering ships before Plan 2 begins and has no engineering prerequisites — it edits `src/network.py`, `src/cpt_data.py`, and `src/inference.py` directly on the existing pgmpy code path. No `NetworkSpec`, no backend abstraction. Plan 3 later lifts the work into the declarative `NetworkSpec` / dual-backend architecture and adds PyMC-native support in Phase 2. The framework write-up has no engineering prerequisites and can proceed in parallel.
>
> **Foundational reference.** This plan applies the *scenario-as-latent BN framework* to the Hormuz network. The framework — five node categories ($\mathcal X, \mathcal M, \mathcal O, S, \mathcal D$) and six edge rules (R1–R6) — is specified in [docs/scenario_bn_framework.md](scenario_bn_framework.md). This plan is the Hormuz **instance** of the framework, not a re-derivation of it. Sections A.1–A.3 below assume framework fluency at the level of that document.
>
> **Related docs.** `docs/master_plan.md` §4 is the in-tree registry of finding IDs and includes M1 / M7 (the findings this plan closes). `docs/03_pymc_integration_plan.md` provides the backend substrate the implementation runs on. Cross-cutting touches in `docs/02_translator_robustification.md` (Plan 2 A1), `docs/04_elicitation_tool_plan.md` (Plan 4 Layer 2), and `docs/05_dashboard_ui_plan.md` (Plan 5 C4) are described in Section C below.
>
> **Status legend.** ⬜ not started · ⏳ in progress · ✅ shipped (with date).

## Executive Summary

The current Hormuz Bayesian network treats `Scenario` as a leaf node with three intermediate-outcome parents (`Energy_Infrastructure_Damage`, `Conflict_Duration`, `Diplomatic_Resolution_Path`) — written $(D, T, P)$ in the math notes — via a 27-column CPT $P(S \mid D, T, P)$. The dashboard review (finding M1) showed that this CPT is mathematically a softmax-like *labelling function* of the three parents, not a generative probabilistic model. Scenario probabilities reported by the dashboard are therefore the expectation of a labelling function under the joint posterior of $(D, T, P)$ — not true Bayesian posteriors over a regime variable.

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

The structural fingerprint of the violation is visible in the entropy of the original CPT $P(S \mid D, T, P)$. On a 3-state distribution, Shannon entropy ranges in $[0, \log_2 3] \approx [0, 1.585]$ bits. The CPT splits cleanly into two regimes:

- **Parent configurations that point cleanly to one scenario** have very low entropy. `none / short / open` (the Stress fingerprint) sits at $H = 0.37$; `severe / long / closed` (the Severe fingerprint) sits at $H = 0.52$. The CPT is near-deterministic on these columns.
- **Parent configurations where the three variables disagree** about which scenario have high entropy, in the $1.1$–$1.5$ bit range (close to the $1.585$-bit maximum). Examples: `moderate / medium / narrowing` ($H = 1.37$), `severe / short / open` ($H = 1.54$).

This **bimodality** — near-deterministic when parents agree on a regime, near-maximum entropy when they disagree — is the diagnostic signature of a labelling rule rather than a generative emission. A genuine $P(O \mid S, \ldots)$ emission CPT would not show this pattern: the regime would modulate but never determine the emission, so column entropies would land in a moderate band rather than splitting into "decided" and "ambiguous" clusters.

Finding M7 — *"Resample-mean vs point-estimate"* — is closed in practical terms by the reframe as a side-effect. The 1–3pp gap between the dashboard's two computation paths comes mostly from the non-linearity of the labelling CPT: point-estimate inference computes $\text{labelling}(E[D, T, P])$, while the resample-mean computes $E_\theta[E[\text{labelling}(D, T, P) \mid \theta]]$, and these differ whenever labelling is non-linear (sharply, at the corners). Once $P(S \mid E)$ becomes a genuine Bayes-rule posterior over a latent regime variable, the labelling step disappears and the gap shrinks to a much smaller residual (~0.1–0.5pp) from general Bayesian non-linearity — the Jensen's-inequality gap between the point-estimate-of-posterior and the posterior-over-point-estimates. That residual is below the precision at which the dashboard would display percentages, so M7 closes without further Plan 5 UI work.

### A.2.5 Why the reversal is necessary: what happens when the client observes an outcome

The §A.2 entropy diagnostic shows the original CPT *behaves like* a labelling rule. The operational consequence of that — and the cleanest demonstration of why the arrows must reverse — comes from walking through a single observation under each topology. Take the concrete case *"the client observes $D = \text{severe}$ — energy infrastructure damage was severe."*

**Under the original $\mathcal O \to S$ topology** — $S$ is a leaf with parents $(D, T, P)$ — the only way to update beliefs about $S$ given $D$ alone is to marginalise the labelling CPT over what the network currently believes about $T$ and $P$:

$$P(S = s \mid D = \text{severe}) \;=\; \sum_{t, p} P(S = s \mid \text{severe}, t, p) \cdot P(T = t, P = p \mid D = \text{severe}).$$

This is the **expectation of the labelling function** under the conditional distribution of the other outcomes. Two operational problems follow:

1. **Context-dependent answer.** The same single observation $D = \text{severe}$ produces a different scenario "probability" depending on what the network currently believes about $T$ and $P$ — because those beliefs are the weights $P(T = t, P = p \mid D = \text{severe})$ in the sum above. There is no posterior on the regime, only an expected label that drifts as upstream beliefs about other outcomes shift.
2. **No Bayes factor.** The quantity stakeholders actually want for governance — *"observing severe damage is X× more likely under Severe_Closure than under Stress_Mitigates"* — is the Bayes factor $\Lambda_{s_1, s_2}(D = \text{severe}) = P(D = \text{severe} \mid S = s_1) / P(D = \text{severe} \mid S = s_2)$. In this topology $D$ is not a child of $S$, so $P(D \mid S)$ is not definable. The governance quantity has no home.

**Under the reversed $S \to \mathcal O$ topology** — $D$ is a child of $S$ with emission CPT $P(D \mid S, M, C)$ — observing $D = \text{severe}$ opens the collider at $D$ and propagates likelihood evidence *back* to $S$ via direct Bayes' rule. After marginalising the unobserved emissions $T, P$ (each of their CPTs sums to 1 over its own state) and the upstream chain except $(M, C)$:

$$P(S = s \mid D = \text{severe}) \;\propto\; \sum_{m, c} P(M = m, C = c) \cdot P(S = s \mid m, c) \cdot P(D = \text{severe} \mid s, m, c).$$

Here $P(M, C)$ is the joint marginal of $(M, C)$ produced by the upstream chain, $P(S \mid m, c)$ is the regime CPT, and $P(D \mid s, m, c)$ is the emission CPT — the standard Bayesian-network posterior factorisation for a latent variable with both upstream parents and downstream emissions. Three things now work:

1. **Genuine Bayesian posterior on the regime**, derived by Bayes' rule. The answer no longer depends on what the network "currently believes" about other outcomes — those marginalise out cleanly because each unobserved-emission CPT is normalised. The observation $D = \text{severe}$ stands on its own as evidence about $S$.
2. **Multiple emissions compose multiplicatively** by independence given $S$. Observing all of $D, T, P$ together gives
   $$P(S = s \mid d, t, p) \;\propto\; \sum_{m, c, \ldots} P(\text{Pa}) \cdot P(s \mid m, c) \cdot P(D = d \mid s, m, c) \cdot P(T = t \mid s, \ldots) \cdot P(P = p \mid s, \ldots),$$
   each observed emission contributing its own likelihood factor. This is the evidence accumulation the client brief calls for, and it composes by independence rather than by labelling-table lookup.
3. **Bayes factors are first-class outputs.** $\Lambda_{s_1, s_2}(D = \text{severe})$ reduces to a parent-averaged column ratio of the emission CPT — directly interpretable as "evidence strength against a regime hypothesis", exactly the governance quantity stakeholders want.

**The load-bearing point.** The client's brief asks for *posterior probabilities over scenarios*. A Bayesian posterior on $S$ is a mathematically well-defined object only when $S$ has a prior (a distribution over its own states) and a likelihood (a way of generating observations given each of its states). Under $\mathcal O \to S$, $S$ has neither: no prior of its own (it is deterministically labelled from its parents) and no likelihood (it is a leaf and cannot generate observations). What the original network produces *can be plotted as a probability* but is mathematically the expectation of a labelling function under the joint posterior of intermediates. Reversing the $S$-arrows is therefore not a stylistic preference: it is the structural prerequisite for the object the client asked for to exist at all.

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

Five structural deltas vs the current network, all enumerated:

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
- **Entropy diagnostic for M1.** Shannon entropy $H(P) = -\sum_i p_i \log_2 p_i$ on a 3-state distribution ranges in $[0, \log_2 3] \approx [0, 1.585]$ bits. The current $P(S \mid D, T, P)$ has $H \in [0.37, 0.52]$ at corner columns and $H \approx 1.37$ at interior columns — the U-shape of a labelling function (framework §10, "labelling-CPT trap").
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

   This procedure is reproducible from the current `src/network.py`. It produces **coherent starting CPTs** that reflect the modeller's existing elicited beliefs projected onto the new topology — a defensible bootstrap that lets Plan 1 ship without waiting for Plan 4 Layer 2 elicitation, and that will be replaced by elicited values once Plan 4 Layer 2 lands. The procedure is implemented as a one-off offline script at `scripts/derive_latent_regime_anchors.py`; its outputs are committed as literal CPT values in `src/cpt_data.py` and consumed at inference time without re-deriving. The script itself is committed alongside its outputs to provide reproducibility and an audit trail for how the v1 CPTs were derived.

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

One implementation hint: translator routing should target emission nodes ($\mathcal O$) and mediator nodes ($\mathcal M$) in roughly equal measure. Headlines about *outcomes* (damage, duration, diplomatic developments) route to $\mathcal O$; headlines about *process* (militia activity, tanker incidents, military posture) route to $\mathcal M$. Both channels propagate to $S$ — the latter via the chain through $\text{Pa}(S)$, the former by opening the collider at the emission node.

### Plan 3 — Inference engine (`03_pymc_integration_plan.md`)

Plan 3 runs after Plan 1 (and Plan 2) and lifts Plan 1's directly-edited files into the declarative `NetworkSpec` / dual-backend architecture:

- **Phase 0** extracts the modified `src/network.py` (carrying Plan 1's latent-regime topology) into a `NetworkSpec`.
- **Phase 1** wraps the existing inference machinery (`BNInferenceEngine` plus Plan 1's Bayes-factor helper) inside `PgmpyBackend`.
- **Phase 2** adds `PymcBackend` for discrete networks, including PyMC-native latent-regime support per §B.1 item 3, validated for parity against Plan 1's pgmpy implementation.
- **Phases 3–4** layer continuous variables (Phase 3) and the Oil_Price migration (Phase 4) on top of the regime topology — direct dependency for $\{D, T, P\}$ if they go continuous, transitive for Oil_Price (which sits in $\mathcal D$, not $\mathcal O$) via the new $P(D \mid S, M, C)$.

Plan 1 has no dependency on Plan 3. The pgmpy `BNInferenceEngine` already supports VE on the latent-regime topology; the three-clamped-inferences pattern is a small helper alongside it, requiring no abstraction layer.

### Plan 4 — Elicitation methodology (`04_elicitation_tool_plan.md`)

Plan 4 Layer 2 (protocol implementations) elicits the new CPTs introduced by Plan 1. The CPTs that experts elicit change shape entirely:

- The old labelling CPT $P(S \mid D, T, P)$ is gone.
- New emission CPTs $P(D \mid S, \ldots)$, $P(T \mid S, \ldots)$, $P(P \mid S, \ldots)$ are elicited from scratch with $S$ as an additional parent.
- A new regime CPT $P(S \mid M, C)$ is elicited.

The elicitation questions become **generative** ("given the regime, what does damage look like?") and **context-conditional** ("given military response and closure status, what's the regime prior?") rather than labelling ("given outcomes, which regime?"). The framework's §6 recipe is the elicitor's playbook.

**Open question for Plan 4 Layer 2:** does expert opinion support a defensible direct $U_3 \to S$ effect that would justify upgrading $\text{Pa}(S)$ from $\{M, C\}$ to $\{U_3, M, C\}$? This is the v2 question the §A.4 blind-spot triage leaves open. If yes, the CPT becomes $P(S \mid U_3, M, C)$ — 18 columns instead of 9. If no, Plan 1's $\{M, C\}$ choice ships unchanged.

Plan 4 Layer 4 (integration) round-trips the elicited CPTs back into `NetworkSpec` and the per-CPT $\kappa$ values become elicited outputs that `PymcBackend` consumes via the provenance pathway.

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
5. **Anchor-derivation procedure.** B.1 item 2's deterministic inversion (run current BN, multiply by current labelling CPT, marginalise + divide) produces all four new CPTs as coherent starting values that reflect the modeller's existing beliefs projected onto the new topology. These are bootstrap values that will be replaced by Plan 4 Layer 2 elicitation.
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
