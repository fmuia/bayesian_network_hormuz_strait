# Bridging Bayesian Networks and Hidden Markov Models: A Framework for Combining Causal Reasoning with Data-Driven Regime Inference

## Executive Summary

This document proposes a framework for integrating two probabilistic models: a **Bayesian network** (BN) encoding expert causal reasoning about geopolitical crises (already implemented in this repository) and a **Hidden Markov Model** (HMM) learning inflation regimes from market data (proposed in a companion document). The BN handles unprecedented events that have no historical data; the HMM handles temporal regime dynamics that experts cannot track in real time.

Four integration mechanisms are presented, in order of increasing ambition:

1. **Structural priors** (Section 3.1.5): The BN's causal beliefs provide informative priors on HMM parameters before estimation.
2. **Transition-matrix covariates** (Section 3.1, Approach A): The BN's scenario posterior enters the HMM's transition matrix via a logistic link, accelerating regime entry when geopolitical evidence warrants it.
3. **Emission modification** (Section 3.3, Approach C): The BN covariate shifts emission distributions, making regimes cause-sensitive without adding latent states.
4. **Hierarchical sub-states** (Section 3.2, Approach B): The HMM's regimes are decomposed into causal sub-types informed by the BN.

The BN also serves as the permanent interface between human expert judgment and the automated HMM — providing structured, auditable overrides rather than arbitrary probability adjustments (Section 6). Both frameworks admit lateral extensions (additional BNs for other risk channels, geographic HMM variants) before the harder hierarchical step (Section 5).

This integration has no direct precedent in the published literature. Section 8 discusses methodological risks (double-counting, calibration mismatch, CPT fragility, structural misspecification) and their mitigations. Section 9 provides 25 verified references.

---

## 1. Introduction and Motivation

This document addresses a specific modelling challenge: how to integrate two distinct probabilistic frameworks — a **discrete Bayesian network** (BN) encoding causal geopolitical reasoning, and a **Bayesian Hidden Markov Model** (HMM) learning inflation regimes from market data — into a coherent system that is stronger than either component alone.

The BN is already implemented (this repository). It models a Strait of Hormuz crisis through 13 causally linked nodes, with expert-elicited conditional probability tables (CPTs), and outputs posterior probabilities over three geopolitical scenarios. The HMM is proposed in a companion document as a Dynamic Scenario Pricing Framework for inflation regime identification, using market and macro indicators to infer latent inflation states in real time.

Both models address a structurally analogous question: *given what we have observed, what is the probability of being in each scenario?* Both use Bayes' theorem to update that answer as new information arrives. Both produce posterior distributions with credible intervals, not just point estimates. (The scenarios themselves differ — geopolitical outcomes for the BN, macroeconomic regimes for the HMM — a distinction explored in Section 3.1.1.)

The surface similarity is real, but it conceals a deep structural difference that determines when each framework is appropriate — and, more importantly, how they compose into something neither can achieve independently.

---

## 2. The Two Frameworks: What They Share and Where They Diverge

### 2.1 Shared Core and Key Divergence

Both models are **probabilistic graphical models over discrete latent states** that maintain posterior distributions over K=3 scenarios, updated by conditioning on observed data. Both propagate uncertainty (the BN through Dirichlet resampling, the HMM through full MCMC posteriors). Both use exact marginalisation (Variable Elimination for the BN, the forward algorithm for the HMM).

The key divergence: **can the regime parameters be learned from historical data?** The HMM says yes — inflation regimes have occurred before (the 1970s oil shocks, the 2021–22 surge), providing decades of market data to learn emission distributions and transition matrices. The BN says no — a sustained Hormuz closure has no historical precedent, so the model encodes expert-elicited causal reasoning instead.

The HMM is a **temporal model with learned parameters** that discovers what each regime looks like but not what causes transitions. The BN is a **causal model with elicited parameters** that explains why a scenario would materialise but has no time axis. They provide complementary signal through partially overlapping but structurally distinct channels: the same event (e.g., a tanker seizure) moves both oil prices (observed by the HMM) and the BN's causal graph, but the BN processes the *causal mechanism* while the HMM processes the *market manifestation*.

### 2.2 Formal Comparison

The HMM is a chain-structured graphical model:

$$
s_1 \to s_2 \to s_3 \to \cdots \to s_T \quad \text{(hidden regime chain)}
$$
$$
\downarrow \quad\;\; \downarrow \quad\;\; \downarrow \qquad\qquad \downarrow
$$
$$
z_1 \quad z_2 \quad z_3 \qquad\quad z_T \quad \text{(observed market data)}
$$

The transition matrix $P$ governs the horizontal arrows. The emission distributions $\mathcal{N}(\mu_k, \Sigma_k)$ govern the vertical arrows. All parameters are learned from data.

The BN is a DAG-structured graphical model with no time axis:

```
Root causes:        Negotiations, Regime_Stability, Sanctions, Mediation
                         ↓
Causal channels:    Proxy_Activity → Tanker_Incidents → Military_Response
                                          ↓                    ↓
Physical state:                    Strait_Closure → Infrastructure_Damage
                                                         ↓
Terminal state:                                      Scenario
```

Evidence enters as observations on any node, and inference propagates through the graph to update the Scenario node. The CPTs are elicited, not learned. There is no Markov chain and no notion of temporal persistence.

---

## 3. Integration Architecture

The BN and HMM can be connected through several mechanisms, each addressing a different aspect of the integration problem. This section presents them in order of increasing ambition: from lightweight coupling (Section 3.1) through intermediate options (Sections 3.2, 3.3) to the fully unified model (Section 3.4). These are **not mutually exclusive** — they address different dimensions and can be composed. The recommended development path (Section 5.4) sequences them by feasibility and value.

### 3.1 Approach A: The BN as an External Signal Injected into the HMM

#### 3.1.1 The Idea

The HMM's transition matrix $P$ governs how likely regime switches are. In the baseline specification, $P$ is estimated from historical data and remains fixed between monthly recalibrations. But a geopolitical event — an escalation in the Strait of Hormuz — should change the transition dynamics themselves. The historical base rate for transitioning from Moderate to High Inflation (perhaps $P(\text{Mod} \to \text{High}) = 0.03$) does not account for unprecedented geopolitical shocks, because such shocks are absent from (or extremely rare in) the training data.

The BN provides exactly the missing signal. When a news headline is processed and the BN posterior shifts toward elevated conflict scenarios, that posterior is used to modify the HMM's transition matrix for the current period.

The standard approach in the TVTP literature (Diebold et al., 1994; Filardo, 1994) parameterises each row of the transition matrix through a logistic (softmax) link function, which guarantees that the adjusted matrix remains a valid stochastic matrix (non-negative entries, rows summing to 1). In our setting, the BN's scenario posterior $\pi^{\text{BN}}$ enters as a covariate in this link:

$$
P_{jk}(t) = \frac{\exp(\alpha_{jk} + \beta_{jk} \cdot x_t)}{\sum_{l} \exp(\alpha_{jl} + \beta_{jl} \cdot x_t)}
$$

where $x_t = g(\pi^{\text{BN}}_t)$ is a scalar covariate derived from the BN posterior (e.g., a weighted escalation score), $\alpha_{jk}$ are the baseline log-odds (encoding the historical transition rates), and $\beta_{jk}$ are coupling parameters that control how strongly the BN signal shifts each transition probability. The logistic link ensures valid probabilities by construction.

An important subtlety: the BN's scenario space (Stress_Mitigates, Prolonged_Conflict, Severe_Closure) is not the same as the HMM's regime space (Low, Moderate, High Inflation). The BN produces geopolitical outcomes; the HMM produces macroeconomic regimes. Severe Closure is a *cause* of High Inflation, not a synonym for it — and the inflationary impact of a Hormuz disruption depends on the broader macro context (oil import dependency, strategic reserves, monetary policy stance). The mapping function $g$ that converts the BN posterior into a transition covariate encodes this cross-domain translation, and it is itself a modelling assumption that should be validated by backtesting against historical episodes where geopolitical shocks produced (or failed to produce) inflationary regime transitions.

#### 3.1.2 What This Achieves

The BN acts as an **event-driven covariate on the transition probabilities** — the same role that the companion HMM proposal envisions for DSGE structural variables at a quarterly frequency (Section 4.5 of that document). The BN operates at a different cadence: not quarterly, but event-driven, firing whenever a significant headline is processed. This creates a natural multi-speed update architecture:

| Speed | Source | What it updates |
|-------|--------|----------------|
| Daily | Market data (breakevens, oil, yields, FX) | HMM filtered probabilities via forward algorithm |
| Event-driven | News headlines → BN | HMM transition matrix via geopolitical adjustment |
| Monthly | Full macro data (CPI, PMI, M2) | HMM parameters via full Bayesian re-estimation |
| Quarterly | DSGE structural model | HMM transition matrix via structural anchor |

The BN and the DSGE play analogous roles — both are slower-moving, structurally informed signals that modify the transition dynamics the fast daily filter operates on. The BN handles geopolitical black swans that have no empirical base rate; the DSGE handles macroeconomic fundamentals.

#### 3.1.3 Implementation Characteristics

**Strengths:**

- Clean separation of concerns. The HMM learns emission parameters from market data; the BN handles causal geopolitical reasoning. The coupling is a single, tunable function.
- Lightweight to implement. No changes to the HMM's internal architecture are required — only a pre-processing step that adjusts $P$ before the daily filter runs.
- Model-agnostic. This approach works with any HMM variant — Gaussian emissions, Student-t emissions, factor-decomposed covariance, different values of $K$ — because it only touches the transition matrix. As the HMM evolves through successive refinements, the BN injection mechanism remains unchanged.
- Immediately useful. The BN is already built and producing scenario probabilities. The coupling function can be prototyped and tested before the HMM is fully operational.

**Limitations:**

- The BN's rich causal structure is compressed into a scalar (or low-dimensional) signal. All 13 nodes, 21 edges, and the full internal posterior are reduced to a nudge on a single transition probability. This is lossy.
- The HMM's "High Inflation" regime remains a monolith. A Hormuz supply shock and a wage-price spiral both land in the same state, but they have different asset return implications, different expected durations, and different policy responses. Approach A cannot distinguish them.

#### 3.1.4 Concrete Example

A headline — "Iran seizes two tankers; US deploys carrier strike group" — shifts the BN posterior to P(Severe\_Closure) = 0.35, P(Prolonged\_Conflict) = 0.45. The mapping function $g$ computes a scalar escalation covariate (e.g., $x_t = 0.45 + 2 \times 0.35 = 1.15$), which enters the logistic link and shifts $P(\text{Mod} \to \text{High})$ upward from its baseline of 0.03. The next day, oil spikes and breakevens jump — the HMM's emission likelihood reinforces the BN's structural signal, and both channels agree that High Inflation is more reachable.

#### 3.1.5 Alternative to Real-Time Coupling: BN as Structural Prior

Approach A feeds the BN's output into the HMM as a real-time covariate. An alternative — closer in spirit to the Del Negro & Schorfheide (2004) DSGE-VAR framework we cite — uses the BN's causal structure to construct **informative priors on the HMM's parameters** before estimation, rather than modifying them at runtime.

The idea: the BN's CPTs encode structural beliefs about what happens when geopolitical conditions change — e.g., that full strait closure with severe damage implies oil above \$120 and long conflict duration. These beliefs translate into prior distributions on the HMM's emission means ($\mu_k$), covariances ($\Sigma_k$), and transition probabilities ($P$):

- Prior on $\mu_{\text{High, oil}}$: positive and large, consistent with the BN's `Oil_Price_Regime` CPT under escalation configurations.
- Prior on $P(\text{High} \to \text{High})$: elevated, consistent with the BN's `Conflict_Duration` CPT implying persistent crises.

The HMM is then estimated from market data with these structurally informed priors. With sufficient data, the likelihood dominates and the priors wash out; with sparse data (the unprecedented-scenario case), the BN's structural beliefs dominate — which is exactly the regime where expert judgment is most needed.

**How this differs from Approach A:** There is no real-time coupling function. The BN's information enters the HMM once, through priors, before estimation. The HMM then runs as a standalone model. This eliminates the double-counting problem (Section 8.1) and the calibration mismatch (Section 8.2) entirely — the BN's information and the market data enter through separate Bayesian channels (prior and likelihood) that are combined coherently.

**The tradeoff:** The BN's information is static — it doesn't update in real time as new headlines arrive. You would need to re-estimate the HMM whenever the BN's assessment changes substantially. This loses the event-driven responsiveness that the TVTP approach provides.

**Practical recommendation:** Use structural priors for the HMM's baseline estimation (encoding the BN's general beliefs about what inflation regimes look like when geopolitical stress is present), *and* use the TVTP coupling for real-time event-driven updating (encoding the BN's assessment of the current geopolitical state). The two mechanisms are complementary: structural priors shape the parameter landscape; real-time covariates navigate within it.

---

### 3.2 Approach B: Hierarchical Extensions to the HMM

#### 3.2.1 The Idea: Adding Depth to the Regime Structure

The standard HMM has a flat regime space: $s_t \in \{\text{Low}, \text{Moderate}, \text{High Inflation}\}$. Each regime has a single set of emission parameters $(\mu_k, \Sigma_k)$. This is adequate for identifying *that* inflation is high, but not for explaining *why* — and the "why" matters for portfolio construction, duration forecasting, and policy response.

The hierarchical extension adds a second layer: each macro regime can be decomposed into **causal sub-states** that determine the mechanism driving the regime. For the High Inflation regime specifically:

$$
s_t = \text{High} \implies c_t \in \{\text{Supply\_Shock}, \text{Demand\_Pull}, \text{Expectations\_Deanchor}\}
$$

Each sub-state produces a distinct emission distribution:

$$
z_t \mid s_t = \text{High},\; c_t = j \;\sim\; \mathcal{N}(\mu_{\text{High}, j},\; \Sigma_{\text{High}, j})
$$

The BN's role in this architecture is to provide the posterior over the sub-state $c_t$, using its causal graph to determine which mechanism is active given the current geopolitical evidence. The BN's internal node posteriors — `Strait_Operationally_Closed`, `Energy_Infrastructure_Damage`, `Oil_Price_Regime` — directly inform which sub-state applies.

**An important caveat on coverage:** The Hormuz BN as currently built provides structural support for only one sub-state: Supply\_Shock. The other sub-states (Demand\_Pull, Expectations\_Deanchor) have no corresponding BN — their priors would need to come from other sources (additional BNs covering monetary policy or expectations dynamics, as discussed in Section 5, or from purely data-driven estimation with weakly informative priors). Until those additional BNs exist, Approach B effectively operates as a two-way decomposition: "supply-shock-driven" (informed by the Hormuz BN) vs "other" (informed by data and generic priors).

**A design choice on sub-state inference:** The sub-state $c_t$ can be handled in two ways. In a **mixture formulation**, $c_t$ is a latent variable marginalised in the forward algorithm alongside the macro regime $s_t$, so the HMM carries uncertainty over which sub-state is active. This is statistically principled but increases the effective state space and estimation difficulty. In a **hard-switch formulation**, $c_t$ is clamped to the BN's MAP assignment at each time step (e.g., Supply\_Shock if the Hormuz BN's escalation covariate exceeds a threshold), and the HMM conditions on that assignment as given. This is simpler but discards the BN's uncertainty about which mechanism is active. The recommended starting point is the hard switch, progressing to the full mixture once the hard-switch version is validated and estimation resources permit.

#### 3.2.2 What This Provides Beyond Approach A

There are four specific gains, each tied to a concrete modelling need.

**1. Causal attribution with portfolio consequences.**

Approach A tells you "we are probably transitioning into high inflation." The hierarchical model tells you "we are transitioning into high inflation *because of a geopolitical supply shock*." This distinction is material for asset allocation. A supply-shock inflation (Hormuz) means oil above \$120, potentially falling real yields as central banks hesitate to tighten into a supply disruption, and strong outperformance of commodity trend-following strategies. A demand-pull inflation means rising real yields, tight labour markets, and a different equity factor profile. Lumping these into one emission distribution averages over mechanisms that produce opposite signals on several key indicators.

**2. Mechanism-specific duration dynamics.**

A Hormuz-driven supply shock might resolve in months if diplomatic channels succeed — the BN's `Diplomatic_Resolution_Path` node captures exactly this conditional structure. A wage-price spiral is self-reinforcing and can persist for years. The hierarchical model lets you assign different expected durations to different sub-states of High Inflation, rather than forcing a single $P(\text{stay in High})$ parameter that averages over all historical causes.

**3. Richer stress testing.**

Approach A can answer "what if P(High) goes up?" The hierarchical model can answer "what if P(High) goes up *because of a Hormuz escalation* versus *because of monetary accommodation*?" — and produce different forward simulation paths for each, because the emission dynamics and duration expectations differ.

**4. A natural home for the BN's Oil\_Price\_Regime node.**

The current BN includes an `Oil_Price_Regime` node that is a parallel consequence of the same geopolitical disruption — it hangs off `Strait_Operationally_Closed` and `Energy_Infrastructure_Damage`, but does not feed into `Scenario`. In the hierarchical model, this node directly informs the conditional emission distribution for oil prices in the Supply Shock sub-state, creating a tight coupling between the causal structure and the market observables.

#### 3.2.3 The Relationship Between Approaches A and B

A critical clarification: Approaches A and B address **different aspects** of the integration and should be understood as complementary layers, not as alternatives.

- **Approach A operates on the arrows *between* HMM states** (the transition matrix). It answers: *should we be more likely to enter the High Inflation regime given the geopolitical evidence?*
- **Approach B operates on the *interior* of an HMM state** (sub-state decomposition). It answers: *given that we are in (or entering) High Inflation, what is the causal mechanism?*

These address different dimensions of the model, though they are not fully independent: both draw on the same BN posterior, and the transition adjustment from Approach A affects how quickly the HMM enters the regime where Approach B's sub-state decomposition becomes active. The two approaches interact through the BN — a strong escalation signal simultaneously increases the transition probability (Approach A) and shifts sub-state selection toward Supply\_Shock (Approach B). This interaction is a feature, not a bug, but it means the two approaches should be calibrated jointly rather than independently tuned.

That said, Approach B by itself has a blind spot: it does not help the HMM *transition into* High Inflation faster when geopolitical evidence warrants it. It only activates once the HMM has already assigned significant probability to the High regime. You still need the Approach A mechanism to handle the fact that unprecedented geopolitical shocks should accelerate regime entry beyond what the historical transition rate implies.

The full architecture with both approaches active:

```
BN processes headline → full posterior over all 13 nodes
                              ↓                        ↓
                     Approach A channel         Approach B channel
                              ↓                        ↓
               Read terminal Scenario node:    Read internal nodes:
               compress to transition          Strait_Closed, Damage,
               adjustment signal               Oil_Price → determine
                              ↓                 sub-type probabilities
               Adjust P(Mod→High)                      ↓
               in the HMM transition           Select emission parameters
               matrix                          μ_{High,Supply_Shock}, Σ_{High,Supply_Shock}
                              ↓                        ↓
               HMM daily filter runs           Asset return implications
               with adjusted P                 are specific to the causal
                                               mechanism
```

The same BN feeds both channels, but through different projections of its posterior. Approach A uses the aggregate Scenario posterior (a summary statistic of the terminal node). Approach B uses the intermediate node posteriors — the internal causal structure that Approach A discards. This means the elicitation effort is shared: one BN, two interfaces.

#### 3.2.4 Estimation Challenges for Hierarchical HMMs

It would be professionally irresponsible to propose hierarchical HMMs without a frank assessment of their estimation difficulty. This is where the framework's ambition meets practical constraints.

**The fundamental problem: data scarcity at the sub-state level.**

A standard K=3 HMM for inflation regimes might have 200–600 monthly observations, with perhaps 40–80 months of "High Inflation" across all historical episodes. This is enough to estimate $\mu_{\text{High}}$ and $\Sigma_{\text{High}}$ reasonably. But decomposing those 40–80 months into sub-states — Supply Shock, Demand Pull, Expectations Deanchoring — means each sub-state may have 10–25 observations. For a d=8 observation vector, estimating a full covariance matrix from 15 observations is ill-conditioned at best and meaningless at worst.

**Identifiability.**

Hierarchical HMMs are notoriously prone to **label-switching** and **non-identifiability** problems. The standard HMM already suffers from these (the model doesn't know which regime is "Low" and which is "High" — you label them post-hoc). Adding a sub-state layer multiplies the problem: you need to identify not just which macro regime is active, but which sub-state within it. Without strong priors or structural constraints, the MCMC sampler may mix poorly between configurations that swap sub-state labels.

**Computational cost.**

The forward algorithm for a flat K=3 HMM scales as $O(T \cdot K^2)$ per likelihood evaluation. A hierarchical model with $K$ macro states and $J$ sub-states within one regime has an effective state space of $K + J - 1$ (the decomposed regime contributes $J$ states instead of 1). For $K=3$ and $J=3$, this gives 5 effective states, so the forward pass scales as $O(T \cdot 25)$ vs $O(T \cdot 9)$ — roughly 3x more expensive. If sub-states are added to all $K$ regimes (effective state space $KJ$), the cost becomes $O(T \cdot (KJ)^2) = O(T \cdot 81)$, which is 9x the flat model. Combined with MCMC (thousands of likelihood evaluations), this can extend runtimes from minutes to hours depending on the specification.

**The practical mitigation: informative priors from the BN.**

This is where the BN earns its keep beyond the Approach A injection. The sub-state emission parameters that cannot be reliably estimated from sparse data can be anchored by **informative priors derived from the BN's causal structure and expert elicitation**. For example:

- The BN's `Oil_Price_Regime` CPT, conditioned on `Strait_Operationally_Closed = full` and `Energy_Infrastructure_Damage = severe`, implies oil prices above \$120 with ~90% probability. This translates directly into an informative prior on $\mu_{\text{High, Supply\_Shock}}$ for the oil component of the emission vector.
- The BN's `Conflict_Duration` CPT, conditioned on breakdown/no mediation/major military response, implies ~85% probability of a long conflict. This informs the prior on the sub-state's persistence parameter.

Without the BN, the hierarchical HMM's sub-state parameters are under-determined by data. With the BN, the priors carry genuine structural information that regularises the estimation. The model documentation (Section 6) already sketches this coupling — the BN posterior fed into PyMC as an informative Dirichlet prior on regime mixture weights.

A note on the BN-to-prior mapping: the BN's nodes have discrete states (e.g., `Oil_Price_Regime` ∈ {`below_90`, `90_to_120`, `above_120`}) while the HMM's emission parameters are continuous (monthly log-returns). Converting a discrete BN posterior over price buckets into an informative prior on a continuous Gaussian mean requires non-trivial design choices: what mean log-return does "above\_120" imply? What variance? These mappings should be treated as explicit modelling assumptions, documented and subject to sensitivity analysis rather than buried in implementation.

**Recommended progression:** The estimation challenges argue strongly for an incremental approach: flat HMM first, then Approach A injection, then hierarchical sub-states only where justified by data and downstream need. The full development path, including lateral extensions, is detailed in Section 5.4.

---

### 3.3 Approach C: BN as Emission Modifier (Time-Varying Emissions)

Approach A modifies *when* the HMM transitions between regimes. Approach B decomposes *what kind* of regime is active. A third option modifies *what each regime looks like* — the emission distributions themselves — as a function of the BN's geopolitical state.

#### 3.3.1 The Idea

Instead of (or in addition to) entering the logistic link on the transition matrix, the BN covariate $x_t$ enters the emission model:

$$
z_t \mid s_t = k \;\sim\; \mathcal{N}(\mu_k + \delta_k \cdot x_t,\; \Sigma_k)
$$

where $\delta_k$ is a vector of regression coefficients controlling how the BN signal shifts the emission mean in each regime. When the BN indicates geopolitical escalation ($x_t$ is high), the High Inflation regime's expected oil return shifts upward, its expected breakeven change increases, and its expected yield change steepens. The Low and Moderate regimes may also be affected, but with different $\delta_k$ vectors — a geopolitical supply shock might increase oil price expectations even in a Moderate regime.

#### 3.3.2 How This Differs from Approaches A and B

**From Approach A:** Approach A says "geopolitical escalation makes regime *entry* more likely." Approach C says "geopolitical escalation makes each regime *look different*." A Hormuz escalation under Approach A increases P(High Inflation) but doesn't change what High Inflation looks like once you're in it. Under Approach C, the same escalation makes High Inflation more extreme — higher expected oil, wider breakevens — even if the transition probability is unchanged.

**From Approach B:** Both address the question "what does High Inflation look like given the underlying cause?" But Approach B answers it through discrete sub-states (Supply_Shock vs Demand_Pull), each with its own fixed emission parameters. Approach C answers it through continuous modulation — the emission parameters slide as a function of the BN covariate, without discrete switching. Approach C is lighter-weight: it adds $d \times K$ parameters (the $\delta_k$ vectors) rather than doubling the effective state space.

#### 3.3.3 When to Use This

Approach C is a natural **intermediate step** between Approach A and the full hierarchical Approach B. If Approach A is too coarse (it only affects transitions, not what happens within a regime) but Approach B is too expensive (full sub-state estimation with scarce data), emission modification provides cause-sensitive emissions without the hierarchical machinery.

It also composes with Approach A: the BN covariate can enter *both* the transition logistic link and the emission mean simultaneously. The transition channel handles "should we enter High Inflation?" while the emission channel handles "how extreme is High Inflation right now?" — two dimensions of the same geopolitical signal.

**Estimation difficulty:** Moderate. The $\delta_k$ parameters interact with the emission likelihood and must be estimated jointly with $\mu_k$ and $\Sigma_k$. This is harder than Approach A (which only touches the transition matrix outside the likelihood) but substantially easier than Approach B (which expands the latent state space). Standard Bayesian regression within the HMM framework; well within PyMC's capabilities.

---

### 3.4 The Theoretical Endpoint: A Joint Model

All preceding approaches treat the BN and HMM as **separate models connected by coupling functions** — the BN runs independently, produces output, and that output is fed into the HMM via covariates, priors, or sub-state selectors. The HMM doesn't know it's talking to a BN.

The fully principled alternative is to embed both in a **single probabilistic model** — a Dynamic Bayesian Network (DBN) where the BN's discrete geopolitical nodes and the HMM's regime chain are joint latent variables, and inference over all of them is performed simultaneously.

#### 3.4.1 What This Would Look Like

At each time step $t$, the model contains:
- The HMM's regime variable $s_t \in \{\text{Low}, \text{Moderate}, \text{High}\}$, following a Markov chain
- The BN's geopolitical node states (e.g., `Tanker_Incidents_t`, `Strait_Closure_t`), following their own temporal dynamics
- Emissions $z_t$ that depend on both $s_t$ and the relevant BN node states

Evidence from headlines enters as observations on the BN's nodes; market data enters as observations on the emissions. Joint inference — marginalising over both the regime chain and the geopolitical nodes — produces a posterior that coherently combines both information sources, with uncertainty propagating correctly across all variables.

This is a DBN, and both standard BNs and HMMs are special cases of DBNs. The unified model would eliminate all the coupling-function design problems (no $g$, no $\beta_{jk}$, no "most active BN" criterion) because the coupling emerges naturally from the joint probability structure.

#### 3.4.2 Why This Is Not Proposed as a Near-Term Step

**Computational cost.** The BN has 13 nodes with 2–3 states each. Marginalising over all of them at every time step, alongside the HMM's regime chain, produces a combinatorial state space that makes MCMC intractable without aggressive approximation (variational inference, structured mean-field, or custom message-passing).

**Temporal elicitation burden.** The BN's nodes would need temporal transition probabilities — how does `Tanker_Incidents` at time $t$ depend on `Tanker_Incidents` at time $t-1$? The current BN has no temporal dynamics; adding them requires eliciting an entirely new layer of CPTs for every node.

**Diminishing returns.** Much of the joint model's benefit can be approximated by the simpler approaches. Structural priors (Section 3.1.5) capture the BN's steady-state beliefs. TVTP coupling (Section 3.1) captures event-driven updating. Emission modification (Section 3.3) captures cause-sensitive regime behaviour. The joint model provides exact coherence, but the incremental approaches provide most of the value at a fraction of the cost.

**Role in the development path:** The joint DBN is best understood as the **theoretical north star** that the incremental steps are converging toward, not as a near-term deliverable. Each intermediate step (structural priors → TVTP → emission modification → hierarchy) captures one more aspect of what the joint model would provide automatically. If the incremental steps eventually prove insufficient, the joint model is where the architecture would need to go — but that decision should be driven by demonstrated limitations of the simpler approaches, not by theoretical elegance.

---

## 4. The Elicitation Problem: Building and Maintaining a Bayesian Network

The technical integration described above presupposes that a well-specified BN exists. In practice, building and maintaining the BN is at least as challenging as the statistical modelling, and it involves cognitive, social, and organisational dimensions that pure software cannot resolve.

### 4.1 What Elicitation Means in This Context

The Hormuz BN has 13 CPTs containing 139 probability columns. Each column is a probability vector filled in by human judgment. The terminal `Scenario` CPT alone has 27 columns of 3 probabilities each — 81 individual numbers, all hand-chosen. The question is: how does one obtain good numbers, and how does one know they are good?

### 4.2 Why Elicitation Is Hard

The difficulty is threefold.

**Combinatorial explosion of assessments.** A node with 3 parents of 3 states each requires 27 conditional probability assessments. Each assessment is itself a distribution over the child's states. One cannot ask an expert for 81 cognitively independent probability judgments. By the twentieth question, answers are anchored on previous responses, and the expert is pattern-matching rather than reasoning from first principles.

**Systematic biases in human probability judgment.** Decades of research in the calibration literature (Kahneman, Tversky, and subsequent work) demonstrate that experts exhibit overconfidence in extreme probabilities, anchoring bias (the first number stated distorts all subsequent assessments), and base-rate neglect. When asked "if sanctions are tightening and the regime is unstable, what is the probability of high proxy activity?", the answer is influenced more by whichever scenario is most cognitively available than by careful probabilistic reasoning.

**Convergence across experts.** If three experts are asked the same CPT question, they will give three different answers. Getting a room to converge on $P(\text{Tanker\_Incidents} = \text{frequent} \mid \text{Proxy\_Activity} = \text{high}, \text{Negotiations} = \text{breakdown}) = 0.60$ versus $0.75$ is a social process as much as a technical one. The difference matters, because it propagates through the network.

### 4.3 Structured Elicitation Protocols

The gold standard in the Bayesian elicitation literature is the **SHELF protocol** (Sheffield Elicitation Framework), developed by O'Hagan and colleagues. SHELF prescribes a multi-phase workflow:

1. **Calibration training.** Before eliciting any CPTs, calibrate experts on probabilistic thinking using questions with known or verifiable answers. This surfaces and partially corrects systematic biases.

2. **Individual assessment.** Each expert independently provides probability judgments before any group discussion. This prevents anchoring on the most vocal participant.

3. **Structured group reconciliation.** Experts share their reasoning — not just their numbers — and discuss disagreements. The facilitator records both the agreed distribution and the spread of individual views. The spread itself is informative: it indicates where genuine uncertainty exists versus where experts simply haven't thought carefully.

4. **Feedback and consistency checking.** After eliciting CPTs, run the full network and show experts the implied scenario probabilities under various evidence configurations. "Your CPT entries collectively imply that under full escalation, the probability of Stress\_Mitigates is 15%. Does that match your overall judgment?" This step catches internal inconsistencies that are invisible when filling in individual CPT entries in isolation.

Step 4 is critical and underappreciated. **The BN itself is the consistency-checking tool.** No human can mentally propagate probabilities through a 13-node graph with 21 edges. The BN can, instantly. This makes the elicitation workflow inherently iterative: elicit CPTs → run inference under various evidence configurations → show experts the implications → adjust → repeat. The Streamlit dashboard already implemented in this repository is directly useful for this purpose.

### 4.4 How the BN Serves as a Communication Device

One of the BN's most significant practical advantages over the HMM is its value for stakeholder engagement:

**The graph IS the conversation.** When an investment committee sees the Hormuz BN diagram, they can immediately engage with the causal structure: "Why does Sanctions\_Trajectory point to US\_Military\_Response? Shouldn't it go through Tanker\_Incidents first?" "I think Mediation should have a stronger influence on Conflict\_Duration." These are substantive debates about causal structure that non-technical stakeholders can participate in meaningfully. The HMM offers no equivalent — a transition matrix and a set of Gaussian emission parameters are not objects that a portfolio manager can interrogate or challenge.

**CPTs are auditable assertions.** When the BN reports $P(\text{Severe\_Closure}) = 0.35$, a stakeholder can ask "why?" and receive a trace through the inference path: "Because we observed frequent tanker incidents, which makes strait closure likely under your assessed CPT, which combined with the long conflict duration gives high probability to severe closure." Every step is a CPT entry that was explicitly discussed and agreed upon. The HMM's filtered probability, by contrast, emerged from a Gaussian likelihood times a transition prior learned from 50 years of data — accurate, perhaps, but not auditable in the same way.

**Scenario narratives are embedded in the graph structure.** The three causal channels in the BN — escalation, military response, diplomatic resolution — map directly to the narrative scenarios that committees already use. This makes the BN a natural scaffolding for structured discussion: "What would have to change for us to move from Prolonged\_Conflict to Stress\_Mitigates?" The BN answers this precisely: the `Diplomatic_Resolution_Path` would need to shift from "narrowing" to "open," which requires either successful negotiations or active third-party mediation. This level of narrative traceability is impossible with the HMM.

### 4.5 How Hard It Is to Converge on a Good Model

Honestly: very hard, and it is never "done." Several specific challenges deserve mention.

**Graph structure is harder than CPT numbers.** Getting the CPT entries approximately right is often easier than getting the graph structure right. Missing an edge — for example, forgetting that `Iranian_Regime_Stability` should influence `Diplomatic_Resolution_Path` — means the model structurally cannot capture a real dependency, regardless of how carefully the numbers are tuned. Adding a spurious edge increases CPT dimensionality and creates phantom dependencies. In practice, graph structure requires multiple rounds of expert review, ideally supplemented by conditional independence tests on whatever partial data may be available.

**The "reasonable range" problem.** For many CPT entries, the difference between 0.60 and 0.75 does not change qualitative conclusions. But for a few critical entries — particularly near the terminal Scenario node — small changes can flip the ordering of scenario probabilities. The Dirichlet sensitivity analysis already implemented in this codebase (`src/sensitivity.py`) is the right tool for identifying which CPT entries matter most. Run the sensitivity analysis, find the entries where the credible intervals are widest, and focus elicitation effort there. Do not spend equal time on every CPT column.

**Temporal drift of expert assessments.** The geopolitical situation changes. A BN elicited in January may have stale priors by June. The CPT for `Sanctions_Trajectory` might assign $P(\text{easing}) = 0.15$, but after a diplomatic breakthrough that prior is wrong. The model documentation (Section 7) already proposes Bayesian learning of CPT parameters — accumulating observation counts to update the Dirichlet priors over time — as the principled solution. This requires a production pipeline that does not yet exist but should be a medium-term development priority.

### 4.6 Tools in the Python and PyMC Ecosystem

| Category | Tool | Role |
|----------|------|------|
| BN construction | **pgmpy** (in codebase) | Discrete BN construction, CPT specification, Variable Elimination inference. No continuous nodes or MCMC. |
| Continuous extensions | **PyMC** | HMM implementation; BN continuous emission layers via `pm.Mixture` and custom log-likelihoods. |
| | **pymc-extras** | `MarginalMixture` and HMM distributions for forward-algorithm marginalisation in NUTS. |
| Structure learning | **pgmpy structure learning** | PC algorithm, Hill Climbing (BIC/BDeu). Validates expert-proposed edges against data. |
| | **CausalNex** | Structure learning with expert edge constraints (must-have/must-not-have). NOTEARS-based. |
| | **bnlearn** (Python) | Simpler API wrapping pgmpy for common structure-learning workflows. |
| Expert elicitation | **SHELF protocol** | Structured group elicitation methodology (R/Excel tools). Calibration, individual assessment, reconciliation. |
| | **Elicitpy** | Python elicitation methods (roulette, quartile) for converting expert beliefs to Dirichlet priors. |
| | **PreliZ** (PyMC) | Interactive prior elicitation; natural fit for CPT column priors within PyMC workflows. |
| Diagnostics | **ArviZ** (PyMC) | Posterior diagnostics (R-hat, ESS), model comparison (`loo`, `compare`). Essential once CPTs become learnable parameters. |
| | `src/sensitivity.py` | Lightweight Dirichlet resampling already in codebase; precursor to full ArviZ workflow. |

---

## 5. Lateral Moves: Expanding Coverage Before Adding Depth

The development path in Section 5.4 presents a vertical progression — flat HMM, then BN injection, then hierarchy. But both frameworks also admit **lateral moves**: expanding the scope of what each model covers, independently, before deepening the integration. These lateral extensions are often lower-risk and higher-impact than the hierarchical step, and they deserve explicit consideration in planning.

### 5.1 Lateral Moves for the BN

The Hormuz BN models one specific geopolitical scenario. But the same methodology — expert-elicited causal graph, LLM translation of news, Streamlit dashboard — can be applied to other event-driven risks that matter for inflation or portfolio construction:

- **A Russia-Ukraine energy disruption BN** with nodes for pipeline flows, European gas storage, sanctions enforcement, and LNG rerouting.
- **A China-Taiwan supply chain BN** with nodes for semiconductor production, shipping lane disruption, US-China sanctions escalation, and tech sector impact.
- **A US fiscal/monetary policy BN** with nodes for Fed communication, fiscal stimulus trajectory, debt ceiling dynamics, and Treasury market stress.

Each BN is a self-contained model with its own graph, CPTs, and dashboard. But crucially, each produces the same type of output: a posterior distribution over a small number of named scenarios. This means multiple BNs can feed into the HMM simultaneously via Approach A — each contributing an event-driven covariate on the transition matrix for a different risk channel.

The architecture for this is a direct extension of the TVTP framework (Diebold et al., 1994; Filardo, 1994): instead of one covariate $x_t$ in the logistic link function, the transition matrix is parameterised with a vector of covariates, one per BN:

$$
P_{jk}(t) = \frac{\exp(\alpha_{jk} + \boldsymbol{\beta}_{jk}^\top \mathbf{x}_t)}{\sum_{l} \exp(\alpha_{jl} + \boldsymbol{\beta}_{jl}^\top \mathbf{x}_t)}, \quad \mathbf{x}_t = \begin{pmatrix} g_1(\pi^{\text{Hormuz}}_t) \\ g_2(\pi^{\text{Russia-Ukraine}}_t) \\ \vdots \end{pmatrix}
$$

Each BN contributes one element of the covariate vector, and the coupling parameters $\boldsymbol{\beta}_{jk}$ determine how strongly each risk channel influences each transition. This is standard TVTP machinery — adding a second BN is adding a second covariate, not changing the model class.

**Why this matters:** Adding a second BN is far easier than adding hierarchy to the HMM. The graph elicitation, translator, and dashboard infrastructure are already built; a new BN reuses all of it. And the multi-covariate TVTP injection requires no change to the HMM's internal architecture. This means the system can grow in topical coverage without waiting for the technically harder hierarchical step.

### 5.2 Lateral Moves for the HMM

The HMM itself can be extended laterally before hierarchy:

- **Geographic variants.** The companion proposal mentions multi-geography extension (US, Europe, Asia, GCC). A separate HMM per region, sharing the same architecture but estimated on local data, expands coverage without architectural changes.
- **Asset-class-specific regimes.** A separate HMM for equity volatility regimes, credit regime switching, or commodity super-cycles, each feeding into portfolio construction through its own channel.
- **Frequency variants.** The companion proposal's dual-speed architecture (daily market filter + monthly re-estimation) is itself a lateral extension — same model, different time scales.

Each of these is a re-estimation of the existing flat HMM on different data, not a structural change. They expand the system's scope without touching the hierarchical machinery.

### 5.3 The Interaction: Multiple BNs into a Single HMM

The most interesting lateral move is combining the two: **multiple BNs, each covering a different event-driven risk, feeding into a single HMM** via Approach A's transition-matrix injection.

This creates a system where the HMM's regime dynamics are influenced by a portfolio of geopolitical risks, not just one. The Hormuz BN might be quiet while the Russia-Ukraine BN is active; the combined effect on the transition matrix reflects the net geopolitical risk environment. This is a richer and more realistic model of how geopolitical events affect inflation regimes — multiple risk channels, each with its own causal structure, contributing to a shared macroeconomic outcome. Technically, it is simply a multi-covariate TVTP model — an established framework in the regime-switching literature, not a new model class.

Importantly, this multi-BN architecture is a prerequisite for, and feeds naturally into, the hierarchical step (Approach B). If the HMM's High Inflation regime is eventually decomposed into sub-states, the relevant sub-state can be selected based on *which BN is most active*. "Most active" requires a precise definition — for instance, the BN whose covariate $g_i(\pi^{\text{BN}}_t)$ currently contributes the largest positive shift to $P(\text{Mod} \to \text{High})$ in the logistic link, or the BN whose covariate has changed most over the recent window. The choice of activation criterion is a design decision that should be validated against historical episodes. Under any reasonable definition: if the Hormuz BN is the dominant driver, the sub-state is Supply Shock; if the monetary policy BN is dominant, the sub-state is Demand Pull. The BN portfolio provides the causal attribution that the hierarchy needs, and each sub-state gets structural prior support from its corresponding BN rather than relying on data alone.

---

## 5.4 Recommended Development Path (Vertical and Lateral)

The integration should be pursued incrementally, with each step delivering standalone value and informing the next. The path includes both vertical moves (adding depth) and lateral moves (adding coverage).

### Step 1: Flat HMM with Learned Parameters

Build and validate the K=3 Bayesian HMM as described in the companion proposal. Gaussian emissions, 8-dimensional monthly observation vector, PyMC implementation with NUTS. Validate against known historical inflation episodes (1970s, 2021–22). Establish the data pipeline, governance layer, and reporting infrastructure.

**Estimation difficulty:** Moderate. This is a well-understood model with established PyMC implementations and sufficient historical data. The main challenge is careful prior specification (particularly Dirichlet priors on the transition matrix rows to encode regime persistence) and convergence diagnostics. Expect 10–30 minutes per MCMC run for the baseline specification.

**Deliverable:** Live inflation scenario probability engine, forward simulation, and controlled stress testing — the full Phase 1 scope. BN-derived structural priors (Section 3.1.5) should be incorporated at this stage to encode the BN's steady-state beliefs about regime characteristics into the HMM's parameter priors.

### Step 2: BN Injection on the Transition Matrix (Approach A)

Add the event-driven BN coupling to the HMM's transition matrix. Implement the logistic link with coupling parameters $\beta_{jk}$ as described in Section 3.1.1. Calibrate the $\beta$ parameters by backtesting: does the BN signal improve the HMM's detection speed for historically known geopolitical-inflation episodes (e.g., the 1990 Gulf War oil shock, the 2022 Russia-Ukraine commodity disruption)?

**Estimation difficulty:** Low. The coupling parameters $\beta_{jk}$ and the mapping function $g$ are design choices that can be calibrated via out-of-sample performance rather than requiring full Bayesian estimation. The BN is already built.

**Deliverable:** Geopolitical early-warning capability. The HMM transitions into High Inflation faster when the BN detects escalation, before market data fully reflects it.

### Step 2b (lateral): Additional BNs for Other Risk Channels

Build one or two additional BNs covering other event-driven risks (e.g., Russia-Ukraine energy disruption, US monetary policy). Reuse the existing graph elicitation, translator, and dashboard infrastructure. Each new BN feeds into the HMM via the same Approach A mechanism, contributing an additional covariate on the transition matrix.

**Estimation difficulty:** Low (for the BNs themselves — same methodology as the Hormuz BN). The multi-covariate logistic link (Section 5.1) requires specifying a mapping function $g_i$ per BN and calibrating the corresponding $\boldsymbol{\beta}_{jk}$ parameters, but this is design and calibration work, not full Bayesian estimation.

**Deliverable:** A system that tracks multiple geopolitical risk channels simultaneously, providing a richer picture of the forces acting on inflation regime dynamics.

### Step 2c (lateral): Geographic or Asset-Class HMM Variants

Re-estimate the flat HMM on data from other regions (Europe, GCC) or for other regime types (equity volatility, credit). Same architecture, different data.

**Estimation difficulty:** Moderate (same as Step 1, repeated). Data availability may vary by geography.

**Deliverable:** Multi-market regime monitoring. Each variant is independently useful and also serves as a validation check (do inflation regimes across geographies correlate as expected?).

### Step 2d (vertical): BN-Modulated Emissions (Approach C)

Add the BN covariate to the HMM's emission model (Section 3.3), so that the geopolitical state affects not just transition probabilities but what each regime looks like. Estimate the $\delta_k$ regression coefficients jointly with the emission parameters.

**Estimation difficulty:** Moderate. Adds $d \times K$ parameters to the emission model. These interact with the likelihood and must be estimated via MCMC, but the model structure remains a standard HMM with regression — no expansion of the latent state space.

**Deliverable:** Cause-sensitive emission distributions. The HMM's High Inflation regime looks different during a geopolitical supply shock than during a demand-pull episode, without requiring discrete sub-states. This is a lighter-weight alternative to Step 3, and may prove sufficient for many applications.

### Step 3: Hierarchical Sub-States for the High Inflation Regime (Approach B)

Decompose the High Inflation regime into J=2 sub-states (supply-shock vs demand-pull) using informative priors from the BN. Estimate sub-state-conditional emission parameters. Validate that sub-states are identifiable (check MCMC convergence, label-switching diagnostics, effective sample size).

**Estimation difficulty:** High. This is the step where the challenges described in Section 3.2.4 become real. The effective state space grows from K=3 to effectively K=3 with J=2 sub-states in one regime (5 effective states). The forward algorithm cost increases, and the MCMC sampler requires careful tuning. Informative priors from the BN are essential to regularise the estimation. Expect to iterate on the prior specification and to encounter convergence difficulties that require reparameterisation or stronger constraints.

Start with J=2, not J=3. Validate thoroughly before expanding. This step is justified only when Phase 2 (regime-conditional asset return modelling) demands causal decomposition of the High Inflation state. If Steps 2b–2c have been completed, the multi-BN portfolio provides a natural mapping to sub-states: the most active BN determines which sub-state is selected.

**Deliverable:** Causal attribution within the High Inflation regime. Different forward simulation paths and asset return implications depending on whether inflation is supply-shock-driven or demand-pull-driven.

### Step 4: Full Multi-Layer Integration

Both Approach A and Approach B operating simultaneously, with multiple BNs feeding both channels. The BN portfolio drives transition matrix adjustment (Approach A). The most active BN's internal node posteriors drive sub-state decomposition (Approach B). BNs are re-elicited or updated periodically through the structured workflow described in Section 4.

**Estimation difficulty:** High, but bounded by the fact that the hierarchy is shallow (two levels) and the sub-state decomposition is applied to only one macro regime. The main ongoing challenge is maintaining the BN portfolio's CPTs as the geopolitical situation evolves — an organisational discipline as much as a technical one.

**Deliverable:** The complete framework: data-driven regime identification (HMM), causal geopolitical reasoning (BN portfolio), and their integration through both transition dynamics and emission decomposition.

---

## 6. The BN as a Bridge Between Manual and Automated Updating

A dimension of the BN's role that deserves explicit emphasis: it spans the full spectrum from **purely manual intervention** to **fully automated data-driven updating**, and every point in between. This makes it uniquely valuable as a transitional architecture — it delivers value today while the HMM is being built, and it continues to add value after the HMM is operational.

### 6.1 The Spectrum

At one extreme, the BN operates as a **structured manual override tool**. An analyst clicks a node in the dashboard, selects a state, and the model propagates the implications. This is pure expert judgment, but disciplined: the analyst cannot set `Tanker_Incidents = frequent` without the model also computing what that implies for `Strait_Operationally_Closed`, `Energy_Infrastructure_Damage`, and ultimately `Scenario`. The BN propagates the causal implications of manual interventions — something a spreadsheet or a committee discussion cannot do.

At the other extreme, the BN operates as an **automated translation layer**. A news headline arrives, the LLM translator maps it to node assignments with confidence distributions, and inference runs without human intervention. This is the daily monitoring workflow: evidence enters automatically, and the model updates continuously.

Between these extremes sit several intermediate modes:

- **Translator with analyst review.** The LLM proposes node assignments; the analyst reviews and selectively accepts or overrides before committing. This is the current dashboard workflow.
- **Batch processing with preview.** Multiple headlines are translated and their combined effect is shown before committing. The analyst exercises judgment on the batch, not on each individual headline.
- **Automated ingestion with exception flagging.** Headlines are translated and committed automatically, but large probability swings or contradictions with recent evidence trigger an alert for human review. The analyst intervenes only when the model detects something unusual.

Each point on this spectrum is a valid operational mode. The BN supports all of them with the same underlying graph structure and inference engine. What changes is the source of evidence (human vs LLM vs automated pipeline) and the governance wrapper (review-before-commit vs commit-then-review vs full automation).

### 6.2 Why This Matters for the HMM Integration

When the HMM is operational, it provides a fully automated, data-driven regime probability — no human input required. But there will always be situations where the committee wants to override or supplement the HMM's output based on information the model cannot process: classified intelligence, informal diplomatic signals, institutional views that resist quantification.

The BN provides the structured channel for these interventions. Rather than asking the committee to directly adjust the HMM's regime probabilities (which would be arbitrary and unauditable), the committee sets evidence on BN nodes — "we believe negotiations have broken down" — and the BN propagates the implications into a transition matrix adjustment (Approach A) or a sub-state decomposition (Approach B). The intervention is causal, traceable, and subject to consistency checking.

This means the BN is not merely a precursor to the HMM that gets retired once the HMM is built. It is the **permanent interface between human judgment and the automated system**. The HMM handles what data can tell us; the BN handles what experts know that data cannot capture. And critically, the BN makes expert overrides auditable: instead of "the committee changed the probability from 30% to 50%," the record shows "the committee set `Negotiations = breakdown` and `Mediation = none`, which the model propagated to a 50% probability of Prolonged Conflict."

### 6.3 Implications for Governance

This bridging role has direct governance implications:

- **Override logging.** Every manual intervention passes through the BN's evidence layer, creating an audit trail of which nodes were set, by whom, when, and why. The HMM proposal (Section 6.4) calls for an override log; the BN provides it structurally.
- **Consistency flagging.** If a committee member sets `Strait_Operationally_Closed = full` and `Energy_Infrastructure_Damage = none`, the BN will accept both inputs but produce a posterior that assigns very low probability to that joint configuration — effectively flagging the inconsistency through its output. The dashboard can surface this as a warning ("your evidence combination has low prior probability under the model's CPTs"), making inconsistent overrides visible and debatable rather than silently accepted.
- **Gradual automation.** As the organisation gains confidence in the automated pipeline, the governance wrapper can be loosened incrementally — from full manual review to exception-based review to full automation — without changing the underlying model. The BN's architecture accommodates all of these modes.

---

## 7. Summary

The Bayesian network and the Hidden Markov Model are not competing approaches. They are complementary tools that address different aspects of the same problem: inferring latent scenario probabilities from heterogeneous evidence.

The HMM learns what each regime looks like from historical market data and filters new observations into regime probabilities in real time. It cannot reason about unprecedented events.

The BN encodes expert causal reasoning about unprecedented scenarios through an auditable graph of conditional dependencies. It does not currently learn from data or model temporal dynamics, though Bayesian updating of its CPT parameters (proposed in the model documentation, Section 7) is a natural extension.

Their integration operates through multiple channels of increasing ambition. The BN can provide structural priors on the HMM's parameters (Section 3.1.5), modify transition dynamics in real time via TVTP covariates (Approach A), modulate emission distributions to make regimes cause-sensitive (Approach C), or decompose regimes into mechanism-specific sub-states (Approach B). These channels are complementary, not alternative, and should be developed incrementally — with structural priors and Approach A delivering immediate value, emission modification (Approach C) providing a lightweight intermediate step, and the full hierarchical extension following only when Phase 2 demands require it. The theoretical endpoint is a joint Dynamic Bayesian Network (Section 3.4), but the incremental approaches capture most of its value at a fraction of the cost.

Beyond the statistical integration, the BN serves as the permanent bridge between manual expert judgment and automated data-driven updating. It provides a structured, auditable channel for human interventions that the HMM cannot accommodate on its own — turning committee overrides from arbitrary probability adjustments into causal, traceable, consistency-checked evidence operations. The BN is not a precursor that the HMM replaces; it is the interface layer that makes the combined system governable.

This integration has no direct precedent in the published literature and carries specific methodological risks — double-counting of information, calibration mismatch between subjective and empirical probabilities, CPT fragility, and structural misspecification of the causal graph. Section 8 discusses these concerns and their mitigations in detail.

---

## 8. Methodological Concerns: What Could Go Wrong

The BN-to-HMM integration proposed in this document has no direct precedent in the published literature. The TVTP literature (Diebold et al., Filardo) uses observable economic covariates, not outputs from a separate causal model. The DSGE-VAR literature (Del Negro & Schorfheide) combines structural and reduced-form models, but both sides are estimated from the same data. Our proposal feeds the output of an expert-elicited model into a data-driven model — a hybrid that raises specific methodological concerns.

### 8.1 Double-Counting of Information

The same event (e.g., a tanker seizure) enters the system twice: through oil prices (HMM emission channel) and through the BN's causal graph (transition adjustment channel). This biases the model toward over-reaction for events that are both newsworthy and market-moving.

**Severity:** Moderate. **Mitigation:** Calibrate the $\beta_{jk}$ parameters via backtesting on historical events where both channels fired (e.g., 2019 Abqaiq attack, 2022 Russia-Ukraine disruption). Optionally, make the coupling state-dependent: weaker when market data already reflects the event, stronger when the BN detects a structural shift not yet priced.

### 8.2 Calibration Mismatch Between BN and HMM

The BN's scenario probabilities are subjective — derived from expert-elicited CPTs, not from data. The HMM's regime probabilities are empirical — derived from historical market data via MCMC. These two probability scales are not commensurable by default. A BN posterior of P(Severe_Closure) = 0.35 does not necessarily mean the same thing as an HMM filtered probability of P(High Inflation) = 0.35. The former is a subjective judgment about an unprecedented event; the latter is a statistical inference from observed data.

**Severity:** High if the mapping function $g$ and coupling parameters $\beta_{jk}$ treat BN probabilities as directly comparable to HMM probabilities. The BN may systematically produce higher or lower probabilities than the HMM for equivalent situations, leading to persistent over- or under-adjustment of the transition matrix.

**Mitigation:** The logistic link adopted in Section 3.1.1 addresses part of this problem — it ensures valid transition probabilities regardless of the covariate scale. But two residual issues remain. First, the mapping function $g$ that converts the BN posterior into a scalar covariate embeds an implicit scale: is $g(\pi) = 1.15$ a large or small signal? The $\beta_{jk}$ parameters absorb this scaling, but their calibration requires historical episodes where the BN would have fired — episodes that may not exist for truly unprecedented scenarios. Second, the BN output is best treated as an ordinal signal ("more or less escalated") rather than a cardinal probability; $g$ should be designed to preserve rank ordering rather than linear proportionality.

### 8.3 CPT Fragility and Error Propagation

The BN's CPTs are the weakest link in the system. They are hand-crafted, never validated against data (for the unprecedented scenarios they model), and their errors propagate into the HMM's transition dynamics. A single poorly calibrated CPT column — say, an overestimate of the probability of strait closure conditional on tanker incidents — would systematically bias the HMM toward the High Inflation regime whenever tanker incidents are reported.

**Severity:** Depends on the sensitivity of the Scenario posterior to individual CPT entries. The existing Dirichlet sensitivity analysis (`src/sensitivity.py`) quantifies this: if the 80% credible interval for P(Severe_Closure) under full escalation evidence spans 0.25 to 0.55, the CPTs are not precise enough to provide a reliable transition adjustment.

**Mitigation:** Three layers of defense. First, the Dirichlet sensitivity analysis identifies which CPT entries the Scenario posterior is most sensitive to, focusing elicitation effort where it matters. Second, as observations accumulate, Bayesian updating of CPT parameters (model documentation Section 7) replaces pure elicitation with data-informed estimates. Third, the coupling parameters $\beta_{jk}$ can be kept small initially and increased as the BN's calibration is validated, limiting the damage from CPT errors.

### 8.4 The LLM Translator as a Noisy Channel

The LLM translator that converts headlines into BN evidence is noisy and potentially biased: it may anchor on dramatic language, overweight escalation signals, map similar headlines inconsistently, or hallucinate causal connections.

**Severity:** Moderate to high, depending on governance wrapper. **Mitigation:** Human review before committing (Section 6). A news memory database (companion next-steps document, Category E) would enable consistency checking against past translations. The accumulated corpus could also be used to evaluate translator calibration over time.

### 8.5 Non-Stationarity of the BN-HMM Relationship

The relationship between geopolitical events and inflation regime transitions changes over time. The 1973 oil embargo produced severe inflation; a similar disruption today, with strategic petroleum reserves and shale capacity, might produce a milder one. The $g$ and $\beta_{jk}$ calibrated on historical episodes may not transfer.

**Severity:** Low short-term, high long-term. **Mitigation:** Periodic backtesting. The DSGE component (companion proposal Section 4.5) provides a structural anchor reflecting current macro conditions.

### 8.6 Identifiability in the Hierarchical Extension

For Approach B specifically, the hierarchical sub-state decomposition introduces identifiability problems beyond those of the flat HMM. With J=2 sub-states within High Inflation and sparse historical data for each, the MCMC sampler may fail to distinguish sub-states — producing posterior distributions that mix between label-swapped configurations (supply-shock vs demand-pull swapped) or that collapse sub-states into a single effective state.

**Severity:** High for the hierarchical step specifically. This is the primary reason for the incremental development path: the hierarchical extension should not be attempted until the flat HMM and Approach A are validated.

**Mitigation:** Strong informative priors from the BN (as discussed in Section 3.2.4). Order constraints on emission means (e.g., $\mu_{\text{oil, supply\_shock}} > \mu_{\text{oil, demand\_pull}}$) to break symmetry. Careful convergence diagnostics (R-hat, effective sample size, trace plots) with ArviZ. And the pragmatic willingness to retreat to the flat model if sub-states are not identifiable from available data.

### 8.7 Structural Misspecification of the BN

Beyond CPT errors (8.3) and translator noise (8.4), the BN's **graph structure itself** can be wrong — missing causal paths (e.g., a cyberattack channel, an insurance market withdrawal that closes the strait economically) or encoding spurious dependencies. This is qualitatively different because it cannot be corrected by adjusting numbers.

**Severity:** Hard to assess — the BN produces internally consistent posteriors regardless of whether its graph matches reality. Sensitivity analysis identifies which CPT entries matter but cannot identify which edges are wrong.

**Mitigation:** Periodic structural review with domain experts. Conditional independence testing against partial data where available. Structure learning algorithms (pgmpy, CausalNex) for nodes with historical data. Most fundamentally, **maintaining multiple BNs with alternative graph structures** for the same scenario — model averaging across structural specifications — provides robustness against the single-graph assumption, at the cost of additional elicitation effort.

---

## 9. References and Related Literature

The framework proposed in this document draws on several established research streams. This section organises key references by the aspect of our architecture they support, with brief summaries of each paper's contribution.

### Regime-Switching Models in Finance

- **Hamilton (1989).** "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2), 357–384. — Foundational paper introducing Markov regime-switching models; develops ML estimation for unobserved regime shifts in GNP growth.
- **Kim & Nelson (1999).** *State-Space Models with Regime Switching.* MIT Press. — Textbook unifying state-space and Markov-switching estimation via both MLE ("Kim filter") and Gibbs sampling.
- **Ang & Bekaert (2002).** "International Asset Allocation With Regime Shifts." *Review of Financial Studies*, 15(4), 1137–1187. — Dynamic portfolio choice under regime switching; shows ignoring regimes is costly for international allocation.
- **Guidolin & Timmermann (2007).** "Asset Allocation under Multivariate Regime Switching." *JEDC*, 31(11), 3503–3544. — Identifies four regimes in joint stock/bond returns; optimal allocations vary substantially across regimes.
- **Ang & Timmermann (2012).** "Regime Changes and Financial Markets." *Annual Review of Financial Economics*, 4, 313–337. — Comprehensive survey of regime-switching models in finance.

### Time-Varying Transition Probabilities (Approach A)

- **Diebold, Lee & Weinbach (1994).** "Regime Switching with Time-Varying Transition Probabilities." In Hargreaves (ed.), *Nonstationary Time Series*, OUP, 283–302. — Proposes transition probabilities that depend on fundamentals via logistic functions; direct theoretical ancestor of our Approach A.
- **Filardo (1994).** "Business-Cycle Phases and Their Transitional Dynamics." *JBES*, 12(3), 299–308. — Leading-indicator-driven TVTPs improve turning-point prediction in business cycles.
- **Bazzi, Blasques, Koopman & Lucas (2017).** "Time-Varying Transition Probabilities for Markov Regime Switching Models." *JTSA*, 38(3), 458–478. — Score-driven (observation-driven) TVTPs; theoretically grounded alternative to parameter-driven specifications.
- **Pouzo, Psaradakis & Sola (2022).** "MLE in Markov Regime-Switching Models with Covariate-Dependent Transition Probabilities." *Econometrica*, 90(4), 1681–1710. — Rigorous asymptotic theory for covariate-dependent transition matrices under general conditions.

### Hierarchical and Nonparametric HMMs (Approach B)

- **Fine, Singer & Tishby (1998).** "The Hierarchical Hidden Markov Model." *Machine Learning*, 32, 41–62. — Original HHMM paper; recursive generalisation of HMMs capturing multi-scale structure.
- **Murphy & Paskin (2001).** "Linear-Time Inference in Hierarchical HMMs." *NeurIPS 14*. — Casts HHMMs as dynamic Bayesian networks, reducing inference from O(T³) to O(T).
- **Fox, Sudderth, Jordan & Willsky (2011).** "A Sticky HDP-HMM." *Annals of Applied Statistics*, 5(2A), 1020–1056. — Adds a persistence parameter to HDP-HMMs, preventing spurious rapid state switching.
- **Johnson & Willsky (2013).** "Bayesian Nonparametric Hidden Semi-Markov Models." *JMLR*, 14, 673–701. — Extends HDP-HMM with explicit duration distributions for non-geometric regime persistence.

### Bayesian Networks for Risk Assessment

- **Koller & Friedman (2009).** *Probabilistic Graphical Models.* MIT Press. — Standard graduate reference covering BN/MRF representation, inference, and learning.
- **Fenton & Neil (2018).** *Risk Assessment and Decision Analysis with Bayesian Networks.* 2nd ed., CRC Press. — Practical guide to BN-based risk models; covers causal reasoning, influence diagrams, sensitivity analysis.
- **Caldara & Iacoviello (2022).** "Measuring Geopolitical Risk." *AER*, 112(4), 1194–1225. — News-based GPR index; shows geopolitical risk foreshadows lower investment and employment.

### Expert Elicitation

- **O'Hagan et al. (2006).** *Uncertain Judgements: Eliciting Experts' Probabilities.* Wiley. — Comprehensive elicitation methodology covering cognitive biases, structured protocols, and practical guidance.
- **Gosling (2018).** "SHELF: The Sheffield Elicitation Framework." In Dias et al. (eds.), *Elicitation*, Springer, 61–93. — Documents the SHELF protocol for structured group expert elicitation via behavioural aggregation.
- **O'Hagan (2019).** "Expert Knowledge Elicitation: Subjective but Scientific." *The American Statistician*, 73(sup1), 69–81. — Argues elicitation can be rigorous and scientific when following structured protocols.
- **Mikkola, Martin et al. (2024).** "Prior Knowledge Elicitation: The Past, Present, and Future." *Bayesian Analysis*, 19(4), 1129–1161. — State-of-the-art survey; argues practical elicitation tools remain underdeveloped.
- **Icazatti, Abril-Pla, Klami & Martin (2023).** "PreliZ: A Tool-Box for Prior Elicitation." *JOSS*, 8(89), 5499. — PyMC-ecosystem package for interactive prior specification.

### Combining Structural and Reduced-Form Models

- **Del Negro & Schorfheide (2004).** "Priors from General Equilibrium Models for VARs." *IER*, 45(2), 643–673. — DSGE-VAR hybrid where a hyperparameter controls structural vs. data-driven weight; the closest precedent for our BN-to-HMM integration.
- **Del Negro & Schorfheide (2006).** "How Good Is What You've Got? DSGE-VAR as a Toolkit." *FRB Atlanta Economic Review*, 91(Q2), 21–37. — Diagnostics for evaluating whether structural restrictions improve forecasting.

### Probabilistic Programming and Empirical Motivation

- **Salvatier, Wiecki & Fonnesbeck (2016).** "Probabilistic Programming in Python using PyMC3." *PeerJ Computer Science*, 2, e55. — PyMC3 framework paper; the implementation platform for the HMM and BN continuous extensions.
- **Neville, Draaisma, Funnell, Harvey & Van Hemert (2021).** "The Best Strategies for Inflationary Times." *JPM*, 47(8), 8–37. — 95-year empirical study showing commodities and trend-following outperform in inflationary regimes; motivates the HMM's indicator selection.
