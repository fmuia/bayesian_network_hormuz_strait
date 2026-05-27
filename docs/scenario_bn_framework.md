# Scenario-as-Latent Bayesian Network Framework

> **Status.** Foundational design pattern.
>
> **Position.** This document is a *foundational reference* for the scenario-modelling pattern used across this codebase. It is the conceptual substrate that [docs/01_latent_regime_plan.md](01_latent_regime_plan.md) (the Hormuz instance) and any future scenario-BN instance build on. It is not a plan with milestones; it is reusable IP.
>
> **Audience.** The data scientist building or auditing a Bayesian network whose top-level question is *"give me posterior probabilities over a small set of named scenarios."*

## 1. The modelling problem this pattern solves

The recurring brief looks like this:

> *"We are tracking how situation $X$ might resolve. We've identified $k$ scenarios $\{s_1, \ldots, s_k\}$ that capture the qualitatively distinct possibilities. As news arrives, tell us how our probabilities over those scenarios shift."*

The naive approach is to build a BN of causal mechanics whose terminal node is the scenario — i.e., the scenario is a leaf, taking the upstream intermediate variables as its parents, and a CPT $P(\text{Scenario} \mid \text{intermediates})$ encodes the mapping from mechanics to named scenario.

This approach produces a number that *looks* like a probability over scenarios but mathematically is the **expectation of a labelling function** under the joint posterior of the intermediates — not a posterior over a scenario variable. The diagnostic signature of the trap is a CPT that is near-deterministic at "corner" parent configurations and only soft at the disagreeing-parent configurations: a labelling-rule fingerprint, not a generative-emission fingerprint.

The pattern below is the alternative. The scenario becomes a **latent regime variable** that *generates* the regime-defining variables (as emissions) and is *caused by* the regime-shaping process context. Inference over the scenario then becomes a genuine Bayesian posterior, derived by Bayes' rule from evidence applied anywhere in the network.

## 2. The five node categories

The framework partitions every node of the network into one of five disjoint sets.

### 2.1 Exogenous drivers ($\mathcal X$)

Root nodes representing causes that the model treats as external inputs. The modeller carries priors $P(X)$ on them but never tries to explain them. Typical examples: macroeconomic baselines, geopolitical context that comes from outside the modelled domain, regulatory regimes.

**Inclusion test.** Would a domain expert volunteer this variable when asked *"what's driving the situation?"* and treat it as an input rather than a consequence? If yes, it belongs in $\mathcal X$.

### 2.2 Process mediators ($\mathcal M$)

Internal causal nodes that propagate $\mathcal X$ forward toward the outcomes. They have parents in $\mathcal X \cup \mathcal M$ and children in $\mathcal M \cup \mathcal O$ (and possibly $S$; see R5). They represent the *machinery* of how the scenario unfolds.

**Inclusion test.** Does this variable describe a process or state that mediates between exogenous drivers and the scenario-defining outcomes? Would a domain expert say *"this is part of how we get from here to there"*? If yes, it belongs in $\mathcal M$.

### 2.3 Definitional outcomes ($\mathcal O$)

A small set of nodes whose **joint configuration constitutes the operational definition** of which scenario is in force. The client's scenario narratives, stripped of rhetoric, are statements about $\mathcal O$.

**Inclusion test.** If I told you the value of every node in $\mathcal O$, could you uniquely (or near-uniquely) name the scenario? If yes, $\mathcal O$ is the right definitional set. If you'd need more variables to distinguish scenarios, expand $\mathcal O$. If a node in $\mathcal O$ doesn't appear in any scenario narrative, it doesn't belong there.

The size of $\mathcal O$ is typically 2–5: large enough to distinguish scenarios, small enough that each emission CPT is elicitable.

### 2.4 Latent scenario ($S$)

A single categorical variable whose states correspond exactly to the client's named scenarios. The client-facing output. The framework treats $S$ as **latent** — its true value is never directly observed; only its emissions ($\mathcal O$) and downstream readouts ($\mathcal D$) are.

By construction, $|S| = k$ where $k$ is the number of named scenarios. (Hierarchical $S$ structures are an extension; see §11.)

### 2.5 Downstream readouts ($\mathcal D$)

Observable consequences of $\mathcal O$ (and sometimes $\mathcal M$). They don't define the scenario but leak information about it via Bayesian back-propagation through $\mathcal O$. Typical examples: prices, indices, observable counts.

**Inclusion test.** Is this variable a *consequence* of the scenario unfolding, observable as a real-time signal, and not part of the scenario definition? If yes, it belongs in $\mathcal D$.

### 2.6 Notes on categorisation

- **Observability is orthogonal to category.** Any node can in principle be observed (via a translator pipeline, manual evidence entry, etc.). The category describes the node's *causal role*, not its observation status.
- **When a node doesn't fit cleanly**, the most common cause is conflating $\mathcal M$ (process) with $\mathcal O$ (definitional). Apply the §2.3 inclusion test. Failing that, apply §2.5 — if the node is a consequence rather than a cause, it's $\mathcal D$.

## 3. The six edge rules

**R1 — DAG.** The combined graph is a DAG. (Standard BN requirement.)

**R2 — Exogenous drivers are roots.** $\text{Pa}(X) = \emptyset$ for every $X \in \mathcal X$.

**R3 — Mediator chain.** Each $M \in \mathcal M$ has $\text{Pa}(M) \subseteq \mathcal X \cup \mathcal M$. Mediators never receive arrows from $\mathcal O$, $S$, or $\mathcal D$.

**R4 — Outcomes are scenario emissions.** Each $O \in \mathcal O$ has $S$ as one parent, plus its standard causal parents from $\mathcal X \cup \mathcal M$:

$$\text{Pa}(O) = \{S\} \cup \text{causal\_upstream}(O), \quad \text{causal\_upstream}(O) \subseteq \mathcal X \cup \mathcal M.$$

This is the move that makes $S$ a generative latent regime: $S$ biases the distribution of $O$ on top of $O$'s upstream causes.

**R5 — Scenario has context-summary parents.** $\text{Pa}(S) \subseteq \mathcal M \cup \mathcal X$ (never $\mathcal O$ or $\mathcal D$). The parent set must satisfy:

- **R5a (d-connection).** Every $V \in \mathcal X \cup \mathcal M$ that has a directed path to any $O \in \mathcal O$ must also have a directed path to some $P \in \text{Pa}(S)$, where the latter path stays within $\mathcal X \cup \mathcal M$.
- **R5b (parsimony).** $\text{Pa}(S)$ is minimal subject to R5a, drawn from the **downstream-most** mediator layer.

**R6 — Downstream readouts are descendants of outcomes.** Each $D \in \mathcal D$ has $\text{Pa}(D) \subseteq \mathcal O \cup \mathcal M \cup \mathcal D$. In v1, $S$ is *not* a direct parent of any $\mathcal D$ node; readouts inherit regime sensitivity transitively via $\mathcal O$. (Adding $S \to D$ edges is a defensible extension; see §11.)

### 3.1 What R5a is really saying

R5a is the rule that prevents the regime posterior from being deaf to upstream evidence. The intuition:

By R4, every $O \in \mathcal O$ has $S$ as a parent. So any path $V \to \cdots \to O \leftarrow S$ has $O$ as a **collider** (two incoming arrows: one from $V$'s side and one from $S$). Under no conditioning, colliders block — so any $V$ whose only paths to $S$ go through $\mathcal O$ is marginally d-separated from $S$. Evidence on $V$ alone does not move $P(S)$.

R5a requires that every such $V$ also has an alternative path to $S$ that goes through $\text{Pa}(S) \subseteq \mathcal M$ — a chain path, all chain nodes, fully active.

**Practical statement.** In the subgraph induced on $\mathcal X \cup \mathcal M$, every node that has a downstream path to $\mathcal O$ must also have a downstream path to some node in $\text{Pa}(S)$.

If R5a fails for some $V$, $V$ is called a **blind spot**: evidence on $V$ alone will not update $P(S)$ until some downstream node ($O$ or descendant) is also observed and opens the collider. Blind spots are not necessarily fatal — see §5 for the triage procedure.

### 3.2 Why R5b prefers the downstream-most layer

Among parent sets satisfying R5a, the one closest to $\mathcal O$ is preferred because:

1. **Sufficient statistic.** Downstream-most mediators are sufficient statistics for the upstream chain w.r.t. the outcomes. They capture everything upstream that matters for the outcomes; adding upstream nodes as additional $S$-parents is mathematically redundant.
2. **CPT parsimony.** Adding more parents inflates $|\text{Pa}(S)|$ and the regime CPT exponentially.
3. **Elicitation tractability.** *"Given the immediate situational context, what's the regime prior?"* is more concrete than *"Given the macro drivers, what's the regime prior?"* The downstream-most nodes are closer to what experts can intuitively condition on.

## 4. Generic topology

```
        X₁ ⋯ Xₙ                       ← 𝒳  exogenous drivers (roots)
        │      │
        ▼      ▼
        M₁ ⋯ Mⱼ                       ← 𝓜  process mediators
        │      │                          (possibly multi-stage)
        ▼      ▼
        Mⱼ₊₁ ⋯ Mₖ                     ← 𝓜  downstream-most mediator layer
        │ ╲   ╱ │                         (Pa(S) drawn from here under R5b)
        │  ╲ ╱  │
        │   S   │                     ← S   latent scenario
        │  ╱ ╲  │
        ▼ ▼   ▼ ▼
        O₁ ⋯ Oₘ                       ← 𝓞  definitional outcomes
        │      │                          (each has S + causal-upstream parents)
        ▼      ▼
        D₁ ⋯ Dₚ                       ← 𝓓  downstream readouts
```

**Three structural signatures of a well-formed network:**

1. **$S$ is in the middle, not the leaf or the root.** Both incoming arrows (from $\text{Pa}(S) \subseteq \mathcal M$) and outgoing arrows (to $\mathcal O$).
2. **$\mathcal O$ nodes are colliders.** Two parents: $S$ (regime) and upstream (causal). Observing $\mathcal O$ opens the collider; not observing it blocks the path between $S$ and the upstream causes.
3. **Every causal chain $\mathcal X \to \mathcal O$ crosses $\text{Pa}(S)$** (in the cases where R5a holds). Stated as a flow property: $\text{Pa}(S)$ is a topological cut between $\mathcal X \cup \mathcal M$ and $\mathcal O$ that the d-connecting paths can exploit.

## 5. Diagnostic procedure

When building or auditing a network against this pattern, run these six steps in order:

1. **Partition.** Assign every node to exactly one of $\mathcal X, \mathcal M, \mathcal O, S, \mathcal D$. If a node fits two or none, the partition is ill-formed; the most common cause is conflating $\mathcal M$ with $\mathcal O$.

2. **Outcome sufficiency.** For each scenario $s_i$, write the scenario narrative as a constraint on $\mathcal O$. If the narratives don't uniquely identify scenarios from $\mathcal O$ alone, expand $\mathcal O$. If a node in $\mathcal O$ doesn't appear in any narrative, demote it back to $\mathcal M$.

3. **Edge rule compliance.** Verify R2–R6 by direct inspection:
   - No incoming edges to $\mathcal X$.
   - No edges from $\mathcal O \cup S \cup \mathcal D$ into $\mathcal X \cup \mathcal M$.
   - Every $O \in \mathcal O$ has $S$ as a parent.
   - $\text{Pa}(S) \subseteq \mathcal M \cup \mathcal X$.

4. **D-connection audit (R5a).** For each $V \in \mathcal X \cup \mathcal M \setminus \text{Pa}(S)$: in the subgraph induced on $\mathcal X \cup \mathcal M$, check whether $V$ has any descendant in $\text{Pa}(S)$. If not, $V$ is a blind spot.

5. **Parsimony audit (R5b).** For each $P \in \text{Pa}(S)$: check whether removing $P$ creates any new blind spot. If not, $P$ is redundant — consider removal (or justify retention on narrative grounds).

6. **Blind-spot triage.** For each blind spot found in step 4, choose:
   - **Accept and document** — acceptable if the blind-spot node rarely arrives as standalone evidence, or always co-arrives with non-blind-spot evidence.
   - **Extend $\text{Pa}(S)$** — add the blind-spot node directly as a parent of $S$.
   - **Add a structural mediator** — introduce a new $\mathcal M$ node that the blind-spot variable feeds into and that then joins $\text{Pa}(S)$.

## 6. Building a network from scratch

For a fresh client engagement, the framework yields a constructive recipe:

1. **Start with the scenarios.** Get the client to define $\{s_1, \ldots, s_k\}$ in concrete narratives. The narratives don't need to be precise yet, but they need to be qualitatively distinct.
2. **Extract $\mathcal O$ from the narratives.** Read the narratives and list every variable that appears. That's your candidate $\mathcal O$. Apply §5 step 2 to refine.
3. **Identify $\mathcal X$.** Ask the client and domain experts what they consider the "inputs" to the situation — what they treat as given rather than as a consequence.
4. **Sketch $\mathcal M$.** Fill in the causal chains from $\mathcal X$ to $\mathcal O$. The exact structure matters less than capturing the qualitative pathways.
5. **Identify $\mathcal D$.** Ask: what observable consequences of $\mathcal O$ do we get to see in real-time as evidence channels?
6. **Pick $\text{Pa}(S)$.** Start with the downstream-most layer of $\mathcal M$ (closest to $\mathcal O$). Run the diagnostic (step 4–6 of §5) to check R5a/R5b. Iterate.
7. **Elicit CPTs.** Standard elicitation pass — but note that emission CPTs ($P(O \mid S, \text{upstream})$) and the regime CPT ($P(S \mid \text{Pa}(S))$) are the framework-specific objects and need explicit attention.
8. **Validate.** Run the full §5 diagnostic one more time before declaring the network ready.

## 7. Worked example: Hormuz

The Hormuz Strait scenario-forecasting network (see [src/network.py](../src/network.py)) instantiates the framework as follows.

| Category | Hormuz nodes |
|---|---|
| $\mathcal X$ | $U_1$ Negotiations, $U_2$ Regime Stability, $U_3$ Mediation, $U_4$ Sanctions |
| $\mathcal M$ | $A$ Militia, $K$ Tankers, $M$ Military, $C$ Strait Closed |
| $\mathcal O$ | $D$ Damage, $T$ Duration, $P$ Diplomatic Path |
| $S$ | Scenario: $\{$Stress_Mitigates, Prolonged_Conflict, Severe_Closure$\}$ |
| $\mathcal D$ | $O$ Oil Price Regime |

Applying the §5 diagnostic with candidate $\text{Pa}(S) = \{M, C\}$:

- **Step 1 (partition).** Clean — all 13 nodes fit one category.
- **Step 2 (outcome sufficiency).** Scenario narratives at [src/network.py:364-377](../src/network.py#L364-L377) read as conjunctions over $(D, T, P)$ → confirmed.
- **Step 3 (R2–R6 compliance).** All six rules hold under the latent-regime topology specified in Plan 1.
- **Step 4 (d-connection audit).** Reachability of $\{M, C\}$ from each $V \in \mathcal X \cup \mathcal M \setminus \{M, C\}$ in the $\mathcal X \cup \mathcal M$ subgraph:

  | $V$ | Path to $\{M, C\}$ | Covered? |
  |---|---|---|
  | $U_1$ | $U_1 \to K \to M$ | ✓ |
  | $U_2$ | $U_2 \to A \to K \to M$ | ✓ |
  | $U_3$ | (no path — $U_3$ only feeds $T$ and $P$, both in $\mathcal O$) | ✗ **blind spot** |
  | $U_4$ | $U_4 \to M$ direct; $U_4 \to A \to K \to M$ | ✓ |
  | $A$ | $A \to K \to M$ | ✓ |
  | $K$ | $K \to M$ direct; $K \to C$ direct | ✓ |

- **Step 5 (parsimony).** $\{M\}$ alone covers $U_1, U_2, U_4, A, K$. $C$ adds redundant coverage for $K$ but contributes direct closure-evidence sensitivity to $S$ and a cleaner elicitation narrative — kept on those grounds.
- **Step 6 (blind-spot triage).** $U_3$ → **accept and document** for v1. Mediation news rarely arrives as isolated evidence; when it does, the translator typically also routes the same headline to soft evidence on $P$, which opens the collider. Whether to add $U_3$ as a direct parent of $S$ in v2 is deferred to Plan 4 Layer 2 elicitation.

Detailed treatment in [docs/01_latent_regime_plan.md](01_latent_regime_plan.md) (the Hormuz instance plan).

## 8. Inference notes

The pattern is friendly to both pgmpy (exact) and PyMC (sampling-based) backends.

### 8.1 pgmpy

Variable elimination computes $P(S \mid E)$ directly for any evidence $E$. No special handling needed for $S$'s intermediate position.

For **Bayes factor extraction** (useful for reporting evidence strength), use the *three-clamped-inferences pattern* — generalising to $k$ inferences for general $|S| = k$:

1. Pre-compute $P_{\text{marg}}(S = s)$ via one VE call with no evidence.
2. For each $s \in S$: clamp $S = s$, run VE with evidence $E$, obtain $P(E, S = s)$.
3. $P(E \mid S = s) = P(E, S = s) \,/\, P_{\text{marg}}(S = s)$.
4. $\Lambda_{s_1, s_2} = P(E \mid S = s_1) \,/\, P(E \mid S = s_2)$ — the evidence strength against the regime hypothesis.

### 8.2 PyMC

Build $S$ as `pm.Categorical("S", p=cpt_table[<indexed by Pa(S) values>])`. If $\text{Pa}(S)$ values are observed, indexing is direct. If latent, NUTS samples / marginalises them. For small $|S|$ (typically $k \leq 5$), **analytic marginalisation** of $S$ inside the categorical step is recommended for speed and numerical stability.

### 8.3 Soft evidence

Soft evidence on any node is mathematically a hard observation on a virtual binary child of that node. Adding such a virtual child to an $\mathcal O$ node opens the collider at the parent $O$, allowing upstream evidence (or evidence routed only to that $O$) to propagate to $S$. The pattern handles soft evidence without special treatment.

## 9. Parameter uncertainty notes

Each CPT in the network is parameterised by a Dirichlet distribution with concentration $\kappa$. Suggested defaults by category:

| Category | CPT | Suggested $\kappa$ | Rationale |
|---|---|---|---|
| $\mathcal X$ | $P(X)$ root priors | 20–50 | Base rates typically well-anchored by historical context. |
| $\mathcal M$ | $P(M \mid \text{Pa}(M))$ | 20–50 | Causal patterns in process mediators are observable in historical analogues. |
| $\mathcal O$ | $P(O \mid S, \text{upstream})$ emissions | 5–15 | Regime-conditional generation is genuinely uncertain; the modeller is making a counterfactual claim per regime. |
| $S$ | $P(S \mid \text{Pa}(S))$ regime CPT | 5–15 | Same regime-conditional uncertainty applies. |
| $\mathcal D$ | $P(D \mid \text{Pa}(D))$ readouts | varies | Often calibratable from external models (e.g., oil-price models); $\kappa$ can be higher. |

These are starting points; elicited replacements (per the standard elicitation pipeline) override per-CPT.

## 10. Common pitfalls

- **Labelling-CPT trap.** Building $\mathcal O \to S$ instead of $S \to \mathcal O$. The model produces a labelling rule, not a posterior. Fix: invert the arrows (the Plan 1 reframe).
- **S-as-root trap.** Building $S$ as a root with no parents. Violates R5a for every upstream node — the regime posterior becomes deaf to upstream evidence. Fix: add $\text{Pa}(S)$ per R5.
- **Over-parented $S$.** Including all of $\mathcal X \cup \mathcal M$ as parents of $S$. Violates R5b parsimony; CPT explodes. Fix: restrict to the downstream-most layer that satisfies R5a.
- **Sloppy $\mathcal O$.** Including process variables in $\mathcal O$ that aren't part of any scenario narrative. Inflates emission elicitation surface without adding identifiability. Fix: apply §5 step 2.
- **Hidden $\mathcal O \to \mathcal M$ edges.** Variables in $\mathcal M$ that turn out to be downstream of $\mathcal O$. Violates R3 and creates a cycle if $\mathcal O \to S$ direction is also wrong. Fix: re-partition.
- **Treating $\mathcal D$ as $\mathcal O$.** Downstream readouts (oil price, indices) are *consequences*, not definitions. Putting them in $\mathcal O$ makes the scenario depend on the price level rather than on the substantive outcome. Fix: keep $\mathcal D$ as descendants only.

## 11. Extensions beyond the static, single-scenario pattern

The framework as stated covers the static, single-scenario case. Three natural extensions:

- **Temporal coupling (HMM).** Add a transition CPT $P(S_t \mid S_{t-1})$. The static framework is the slice at a single $t$; the HMM extension layers temporal dynamics on top, coupling $S$ to itself across time. See [docs/bn_hmm_integration.md](bn_hmm_integration.md) for the integration with the inflation HMM workstream.
- **Hierarchical scenarios.** Replace $S$ with $S_{\text{coarse}} \to S_{\text{fine}}$ where the coarse scenario states are partitioned into refinements. Useful when the client has both top-level scenarios and within-scenario sub-cases.
- **Regime-modulated readouts ($S \to \mathcal D$).** Adding direct arrows from $S$ to $\mathcal D$ when downstream consequences are regime-conditional *even after* controlling for $\mathcal O$. Requires a separate elicitation pass for the modulated emissions.
- **Multi-regime joint inference.** When several latent regimes coexist (e.g., political regime + economic regime), generalise to a vector $\vec S$ with category-specific emissions. R5a is checked separately for each component.

## Appendix — Math reference

**Bayes factor.** Given two scenarios $s_1, s_2$:

$$\Lambda_{s_1, s_2} = \frac{P(E \mid S = s_1)}{P(E \mid S = s_2)}.$$

Composes by independence: $\Lambda(E_1, E_2) = \Lambda(E_1) \cdot \Lambda(E_2)$ if $E_1, E_2$ are conditionally independent given $S$.

**D-separation rules.**
- Chain $A \to B \to C$ and fork $A \leftarrow B \to C$: $B$ blocks when conditioned on, otherwise active.
- Collider $A \to B \leftarrow C$: opposite — blocks by default, opens when $B$ or any descendant of $B$ is conditioned on.

**Regime posterior factorisation.** With $E_{\text{up}}$ denoting evidence upstream of $S$ (in $\mathcal X \cup \mathcal M$) and $E_{\text{down}}$ denoting evidence on $\mathcal O \cup \mathcal D$:

$$P(S = s \mid E) \;=\; \frac{P(S = s, E)}{P(E)}, \qquad P(S = s, E) \;=\; \sum_{\pi \in \text{Pa}(S)} P(\pi, E_{\text{up}}) \cdot P(S = s \mid \pi) \cdot P(E_{\text{down}} \mid S = s, \pi, \ldots).$$

Exact inference (variable elimination) computes this sum directly. The clamped-inferences pattern (§8.1) extracts each term for Bayes factor reporting.

---

**End of framework.** Companion: [docs/01_latent_regime_plan.md](01_latent_regime_plan.md) (Hormuz instance).
