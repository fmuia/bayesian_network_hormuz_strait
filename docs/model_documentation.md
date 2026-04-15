# Strait of Hormuz Bayesian Network: Model Documentation

## Overview

This project builds a **discrete Bayesian network** that outputs probability distributions over three geopolitical/financial scenarios for the Strait of Hormuz. News headlines are translated into node observations by an LLM; exact inference propagates those observations through the graph; and the dashboard displays updated scenario probabilities with uncertainty bands.

The document walks through every layer of the model: how the graph is defined, how each CPT encodes domain reasoning, how inference works step by step, how uncertainty is quantified, and how the system can be extended to continuous variables, richer priors, and daily sequential updating.

---

## 1) Building the graph: `STATES`, `EDGES`, and the DAG

### 1.1 The state vocabulary (`STATES`)

Every node in the network has a fixed, finite set of named states, collected in the `STATES` dictionary (`src/network.py`, line 19). This dictionary is the **single source of truth** used everywhere: CPT construction, evidence validation, translator schema, dashboard labels.

Each key is a node name; each value is the ordered list of states for that node:

| Node | States | Role |
|------|--------|------|
| `US_Iran_Negotiations` | `success`, `stalled`, `breakdown` | Root driver |
| `Iranian_Regime_Stability` | `stable`, `pressured`, `unstable` | Root driver |
| `Third_Party_Mediation` | `active`, `none` | Root driver |
| `Sanctions_Trajectory` | `easing`, `status_quo`, `tightening` | Root driver |
| `Iranian_Proxy_Activity` | `low`, `elevated`, `high` | Intermediate |
| `Tanker_Incidents` | `none`, `isolated`, `frequent` | Intermediate |
| `US_Military_Response` | `none`, `limited`, `major` | Intermediate |
| `Strait_Operationally_Closed` | `no`, `partial`, `full` | Intermediate |
| `Energy_Infrastructure_Damage` | `none`, `moderate`, `severe` | Intermediate |
| `Conflict_Duration` | `short`, `medium`, `long` | Intermediate |
| `Diplomatic_Resolution_Path` | `open`, `narrowing`, `closed` | Intermediate |
| `Oil_Price_Regime` | `below_90`, `90_to_120`, `above_120` | Intermediate |
| `Scenario` | `Stress_Mitigates`, `Prolonged_Conflict`, `Severe_Closure` | Terminal |

The four **root** nodes have no parents — they represent exogenous geopolitical conditions. The eight **intermediate** nodes form the causal transmission mechanism. The single **terminal** node `Scenario` aggregates everything into the three outcomes the model is designed to forecast.


### 1.2 The edge list (`EDGES`)

`EDGES` (line 35) is a list of `(parent, child)` tuples that defines the directed acyclic graph. Every arrow encodes a **direct conditional dependence** — it means the child's probability distribution changes depending on the parent's state.

The full edge set, grouped by the causal story they encode:

**Escalation channel** (regime instability and sanctions drive proxy activity, which drives incidents):

- `Iranian_Regime_Stability` $\rightarrow$ `Iranian_Proxy_Activity`
- `Sanctions_Trajectory` $\rightarrow$ `Iranian_Proxy_Activity`
- `Iranian_Proxy_Activity` $\rightarrow$ `Tanker_Incidents`
- `US_Iran_Negotiations` $\rightarrow$ `Tanker_Incidents`

**Military response** (incidents and sanctions posture drive US response):

- `Tanker_Incidents` $\rightarrow$ `US_Military_Response`
- `Sanctions_Trajectory` $\rightarrow$ `US_Military_Response`

**Physical disruption** (incidents plus military action drive strait closure and infrastructure damage):

- `Tanker_Incidents` $\rightarrow$ `Strait_Operationally_Closed`
- `US_Military_Response` $\rightarrow$ `Strait_Operationally_Closed`
- `US_Military_Response` $\rightarrow$ `Energy_Infrastructure_Damage`
- `Strait_Operationally_Closed` $\rightarrow$ `Energy_Infrastructure_Damage`

**Conflict dynamics** (negotiations, mediation, and military response drive duration and diplomacy):

- `US_Iran_Negotiations` $\rightarrow$ `Conflict_Duration`
- `Third_Party_Mediation` $\rightarrow$ `Conflict_Duration`
- `US_Military_Response` $\rightarrow$ `Conflict_Duration`
- `US_Iran_Negotiations` $\rightarrow$ `Diplomatic_Resolution_Path`
- `Third_Party_Mediation` $\rightarrow$ `Diplomatic_Resolution_Path`
- `Iranian_Regime_Stability` $\rightarrow$ `Diplomatic_Resolution_Path`

**Macro-financial transmission** (physical disruption drives oil price regime):

- `Strait_Operationally_Closed` $\rightarrow$ `Oil_Price_Regime`
- `Energy_Infrastructure_Damage` $\rightarrow$ `Oil_Price_Regime`

**Scenario synthesis** (damage, duration, diplomacy drive which strategic scenario materializes):

- `Energy_Infrastructure_Damage` $\rightarrow$ `Scenario`
- `Conflict_Duration` $\rightarrow$ `Scenario`
- `Diplomatic_Resolution_Path` $\rightarrow$ `Scenario`

The graph tells us what the model considers relevant causal channels, and — equally important — what it considers **conditionally independent**. For example, `Oil_Price_Regime` does not feed into `Scenario` directly. Scenario is determined by the physical/diplomatic state, not the oil price itself; the oil price is a parallel consequence of the same disruption.


### 1.3 The joint distribution factorization

The graph structure implies that the joint probability over all 13 variables factorizes as:

$$
P(\text{all nodes}) = \prod_{\text{node } X} P(X \mid \text{parents of } X)
$$

Written out for this specific network:

$$
P(\text{Negot}) \cdot P(\text{Regime}) \cdot P(\text{Mediation}) \cdot P(\text{Sanctions})
$$
$$
\cdot\; P(\text{Proxy} \mid \text{Regime}, \text{Sanctions})
$$
$$
\cdot\; P(\text{Tankers} \mid \text{Proxy}, \text{Negot})
$$
$$
\cdot\; P(\text{Military} \mid \text{Tankers}, \text{Sanctions})
$$
$$
\cdot\; P(\text{Strait} \mid \text{Tankers}, \text{Military})
$$
$$
\cdot\; P(\text{Damage} \mid \text{Military}, \text{Strait})
$$
$$
\cdot\; P(\text{Duration} \mid \text{Negot}, \text{Mediation}, \text{Military})
$$
$$
\cdot\; P(\text{Diplo} \mid \text{Negot}, \text{Mediation}, \text{Regime})
$$
$$
\cdot\; P(\text{Oil} \mid \text{Strait}, \text{Damage})
$$
$$
\cdot\; P(\text{Scenario} \mid \text{Damage}, \text{Duration}, \text{Diplo})
$$

Each factor in this product is one CPT. This is the complete probabilistic specification of the model.

---

## 2) How CPTs are built: the `_cpd` function and each table

### 2.1 The `_cpd` helper

The function `_cpd(var, parents, table)` (line 60) builds a pgmpy `TabularCPD` object. It takes:

- `var`: the child node name (string)
- `parents`: ordered list of parent node names
- `table`: a dictionary mapping parent-state tuples to probability vectors over the child's states

For a root node, `parents` is empty and `table` has a single key `()`.

Inside `_cpd`:

1. It looks up child states from `STATES[var]` and parent states from `STATES[p]` for each parent.
2. It generates all parent-state combinations in **pgmpy canonical order** (last parent varies fastest — like nested loops where the rightmost index increments first).
3. For each combination, it reads the probability vector from `table` and validates that it sums to 1.
4. It packs the values into a 2D array where rows = child states, columns = parent configurations.
5. It returns a `TabularCPD` with metadata (variable cardinalities, state names).

The canonical ordering is important: pgmpy expects columns in this exact order, and `_cpd` handles the mapping so that the `table` dictionary can be written in any order for readability.


### 2.2 Root node CPTs (the priors)

Root nodes have no parents. Their CPTs are unconditional probability distributions that represent the **base-rate belief** about each driver before any evidence arrives.

**`CPD_NEGOTIATIONS`** — $P(\text{US\_Iran\_Negotiations})$:

| success | stalled | breakdown |
|---------|---------|-----------|
| 0.20 | 0.55 | 0.25 |

Reasoning: most negotiations stall; outright success is the minority outcome; breakdown is a meaningful tail risk.

**`CPD_REGIME`** — $P(\text{Iranian\_Regime\_Stability})$:

| stable | pressured | unstable |
|--------|-----------|----------|
| 0.30 | 0.50 | 0.20 |

Reasoning: under sanctions the regime is most often pressured; full instability is rarer but non-trivial.

**`CPD_MEDIATION`** — $P(\text{Third\_Party\_Mediation})$:

| active | none |
|--------|------|
| 0.45 | 0.55 |

Reasoning: mediators (Oman, Qatar, EU) are active roughly half the time during Gulf flare-ups.

**`CPD_SANCTIONS`** — $P(\text{Sanctions\_Trajectory})$:

| easing | status_quo | tightening |
|--------|------------|------------|
| 0.15 | 0.55 | 0.30 |

Reasoning: status quo dominates; tightening is more likely than easing in the current environment.

These four vectors are the starting point. When no evidence has been entered, the entire model's output flows from these root priors through the CPTs downstream.


### 2.3 Intermediate node CPTs

Each intermediate CPT encodes: *given a specific combination of parent states, what is the probability of each child state?*

The number of columns in each CPT equals the product of parent cardinalities. For example, `Iranian_Proxy_Activity` has parents `Iranian_Regime_Stability` (3 states) and `Sanctions_Trajectory` (3 states), so its CPT has $3 \times 3 = 9$ columns.

I'll walk through two CPTs in detail to show the pattern.

#### CPT: `Iranian_Proxy_Activity` | `Regime`, `Sanctions`

This table has 9 columns (3 regime states $\times$ 3 sanctions states). Each column is a probability vector $[P(\text{low}), P(\text{elevated}), P(\text{high})]$.

The domain logic: a stable regime under easing sanctions has little incentive for proxy aggression; an unstable regime under tightening sanctions has strong incentive to lash out externally.

Selected columns to illustrate the gradient:

| Regime | Sanctions | $P(\text{low})$ | $P(\text{elevated})$ | $P(\text{high})$ |
|--------|-----------|------:|------:|------:|
| stable | easing | 0.85 | 0.13 | 0.02 |
| pressured | status_quo | 0.30 | 0.50 | 0.20 |
| unstable | tightening | 0.05 | 0.25 | 0.70 |

Read the first row as: "If the regime is stable and sanctions are easing, proxy activity is low with 85% probability." Read the last row as: "If the regime is unstable and sanctions are tightening, proxy activity is high with 70% probability."

#### CPT: `Tanker_Incidents` | `Proxy_Activity`, `Negotiations`

This table has $3 \times 3 = 9$ columns. Each column is $[P(\text{none}), P(\text{isolated}), P(\text{frequent})]$.

The domain logic: tanker incidents track proxy activity but are dampened by successful negotiations (back-channel restraint signals).

| Proxy | Negotiations | $P(\text{none})$ | $P(\text{isolated})$ | $P(\text{frequent})$ |
|-------|-------------|------:|------:|------:|
| low | success | 0.92 | 0.07 | 0.01 |
| elevated | stalled | 0.30 | 0.55 | 0.15 |
| high | breakdown | 0.02 | 0.23 | 0.75 |

The extreme column — high proxy activity with breakdown — concentrates 75% probability on frequent incidents.

#### Remaining CPTs (summary of sizes and logic)

| CPT | Parents | Columns | Domain logic |
|-----|---------|---------|-------------|
| `CPD_MILITARY` | Tankers, Sanctions | 9 | Tightening posture + incidents → willingness to use force |
| `CPD_STRAIT` | Tankers, Military | 9 | Closure follows either denial-of-passage or kinetic exchanges |
| `CPD_DAMAGE` | Military, Strait | 9 | Damage requires kinetic action; full closure often co-occurs |
| `CPD_DURATION` | Negot, Mediation, Military | 18 | Success + mediation → short; breakdown + major → long |
| `CPD_DIPLO` | Negot, Mediation, Regime | 18 | Open path needs willing counterparties + stable regime |
| `CPD_OIL` | Strait, Damage | 9 | Full closure + severe damage → price spike |
| `CPD_SCENARIO` | Damage, Duration, Diplo | 27 | Maps joint physical/diplomatic conditions to three scenarios |

The `CPD_SCENARIO` table (27 columns) is the largest. It encodes the three scenario narratives structurally:

- **Stress Mitigates**: low/no damage, short duration, open diplomatic path
- **Prolonged Conflict**: moderate damage, long duration, narrowing/closed path
- **Severe Closure**: severe damage, long duration, closed path

For example, `(damage=severe, duration=long, diplo=closed)` yields $[0.01, 0.09, 0.90]$: 90% weight on Severe Closure.


### 2.4 Assembling the network: `build_network()`

`build_network()` (line 380) does three things:

1. Creates a `DiscreteBayesianNetwork` from `EDGES` — this builds the DAG structure.
2. Attaches all 13 CPTs via `net.add_cpds(...)`.
3. Calls `net.check_model()` — pgmpy validates that every node has exactly one CPT, parent cardinalities match, and all CPT columns sum to 1.

If `check_model()` passes, the network is a valid, complete probabilistic model ready for inference.

---

## 3) Inference: from evidence to posteriors

### 3.1 What inference computes

Given evidence $E = e$ (a subset of nodes fixed to specific states), the query is:

$$
P(S = s \mid E = e) = \frac{\sum_{z} P(S=s, Z=z, E=e)}{\sum_{s'}\sum_{z} P(S=s', Z=z, E=e)}
$$

where $Z$ denotes all unobserved (non-evidence, non-query) nodes. The numerator sums the joint probability over all configurations of hidden nodes consistent with scenario state $s$ and evidence $e$. The denominator normalizes across scenario states.

The same formula applies for querying any node, not just `Scenario`.


### 3.2 Variable elimination (the algorithm)

pgmpy's `VariableElimination` performs exact inference. Conceptually:

1. Start with the set of all CPT factors.
2. For each evidence node, reduce its factor to the observed state (zero out non-matching rows).
3. For each hidden variable $Z_i$ (not query, not evidence), multiply all factors that mention $Z_i$, then sum over $Z_i$'s states. This eliminates $Z_i$ from the factor product.
4. After eliminating all hidden variables, the remaining factor is over the query variable only. Normalize it to get the posterior.

The order in which hidden variables are eliminated affects computational cost but not the result. For a network of this size (13 nodes, max 3 states each), elimination is effectively instant.


### 3.3 The `BNInferenceEngine` class

`BNInferenceEngine` (`src/inference.py`) wraps pgmpy's inference in a stateful API:

- **`__init__`**: takes an optional pre-built network (defaults to `build_network()`), creates a `VariableElimination` engine, initializes an empty evidence dict.

- **`update_evidence(evidence)`**: merges new `{node: state}` pairs into the stored evidence dict. Validates every node name against `STATES` and every state value against `STATES[node]`. If the same node is set again, the new value overwrites the old one.

- **`clear_evidence()`**: resets evidence to empty — returns the model to the prior.

- **`get_scenario_probabilities()`**: queries $P(\text{Scenario} \mid E)$ under current accumulated evidence. Calls pgmpy's `query(["Scenario"], evidence=self._evidence)`.

- **`get_prior_probabilities()`**: queries $P(\text{Scenario})$ with no evidence (regardless of stored evidence), for comparison purposes.

- **`get_node_marginal(node)`**: queries $P(\text{node} \mid E)$. If the node is itself in evidence, returns a delta distribution (1.0 on the observed state, 0.0 elsewhere) directly, without calling pgmpy.

- **`_distribution(factor, node)`**: static helper that converts a pgmpy `DiscreteFactor` to a `{state: probability}` Python dict.

The engine does **not** rebuild the network when evidence changes. pgmpy's VE handles evidence as a per-query argument, so updating evidence is just updating a Python dictionary and requerying.

---

## 4) News-to-evidence translation

### 4.1 The problem

A raw headline like *"Fourth tanker incident in two weeks; insurers raise war-risk premia"* is unstructured text. The model needs structured evidence: `{Tanker_Incidents: "frequent"}`.

### 4.2 How `src/translator.py` solves it

The translator sends the headline to an LLM (Claude or OpenAI) with a **system prompt** that describes:

- the three scenarios and their narratives
- every node name and its allowed states (read from `STATES`)
- instructions to output a JSON object with `assignments` (list of `{node, state, reason}`) and an `overall_rationale`
- constraints: only assign nodes the headline directly speaks to; do not set the `Scenario` node (it is terminal, inferred not observed)

The LLM response is validated in two stages:

1. **Schema validation**: OpenAI uses `strict` JSON mode; Claude output is parsed with `_extract_json_block`.
2. **Domain validation** (`_validate_payload`): each `node` must be in `STATES`; each `state` must be in `STATES[node]`; any `Scenario` assignments are silently dropped.

The result is a `TranslatorResult` containing validated `TranslatorAssignment` objects, each with a node, state, and reason string. The `.as_evidence_dict()` method flattens these to a `{node: state}` dict suitable for `update_evidence`.


### 4.3 How evidence flows in the dashboard

In `app/dashboard.py`:

1. User enters a headline (or clicks an example).
2. Dashboard calls `translate_headline(...)`, streaming progress to a status widget.
3. On success, `_append_observation(...)` creates an `Observation` record (day, headline, assignments, rationale, source) and appends it to `st.session_state.observations`.
4. `_merged_evidence()` iterates over all observations in order; for each observation it calls `.update()` on a merged dict. **Latest assignment wins** per node.
5. `engine.update_evidence(merged)` is called, and `engine.get_scenario_probabilities()` produces the new posterior.
6. For the day-by-day probability evolution chart, the dashboard replays this process cumulatively for each day.

---

## 5) Parameter uncertainty: Dirichlet resampling

### 5.1 The problem

Every number in every CPT is an elicited expert judgment, not a data estimate. How sensitive are the scenario probabilities to those specific numbers?

### 5.2 What `src/sensitivity.py` does

The module treats each CPT column vector $\theta_j$ as the **mean** of a Dirichlet distribution and draws perturbed versions:

$$
\theta_j^{(m)} \sim \mathrm{Dirichlet}\bigl(c \cdot \theta_j + \epsilon\bigr)
$$

where $c$ is the **concentration** parameter (default 20) and $\epsilon = 10^{-6}$ prevents zero-alpha entries.

The Dirichlet distribution over a $K$-dimensional probability vector with parameter $\alpha = (\alpha_1, \ldots, \alpha_K)$ has the property:

- $\mathbb{E}[\theta_k] = \alpha_k / \sum_i \alpha_i$, so using $\alpha = c \cdot \theta_j$ means the expected value of the resample equals the original CPT column.
- The variance of each component decreases as $c$ grows — larger $c$ means resampled columns stay closer to the original values.

The function `_resample_cpd` (line 21) iterates over each column of a CPT and draws a new column from the corresponding Dirichlet. `_resampled_network` (line 39) applies this to every CPT in the network, building a fresh `DiscreteBayesianNetwork` with perturbed parameters.

`scenario_credible_intervals` (line 48) does:

1. Build or receive a base network.
2. For $m = 200$ iterations: resample all CPTs, run VE on the perturbed network under the current evidence, collect scenario posteriors.
3. Compute the mean and the central 80% credible interval (10th and 90th percentiles) for each scenario state.

This gives the uncertainty bands displayed in the dashboard as "[80% CI: X% – Y%]".

### 5.3 Interpreting concentration

- $c = 20$ (current default): moderate confidence. Each column behaves as if backed by roughly 20 pseudo-observations. Resampled columns are noticeably but not wildly different from originals.
- $c = 5$: low confidence. Columns can differ substantially, producing wider credible intervals.
- $c = 100$: high confidence. Columns barely move, intervals are tight.

The choice of $c$ encodes how much you trust the elicited numbers. It is a subjective parameter.

---

## 6) Extension: continuous variables with PyMC

### 6.1 What the current model cannot do

All nodes are discrete categories. This means market observables (returns, spreads, freight rates) cannot be used directly — they would need to be binned into categories, losing information.

### 6.2 The hybrid model idea

Keep the discrete latent geopolitical structure but add **continuous observation layers** that depend on it.

Suppose `Oil_Price_Regime` (currently discrete: `below_90`, `90_to_120`, `above_120`) is recast as a latent categorical variable $Z_t \in \{0, 1, 2\}$. Attach a continuous observable:

$$
Y_t \mid Z_t = k \sim \mathcal{N}(\mu_k, \sigma_k)
$$

where $Y_t$ could be daily Brent log-return or freight rate change, $\mu_k$ and $\sigma_k$ are regime-specific location and scale parameters with their own priors:

$$
\mu_k \sim \mathcal{N}(0, 1), \qquad \sigma_k \sim \text{HalfNormal}(1)
$$

Now observed market data $Y_{1:T}$ provides **likelihood signal** about which latent regime is active, combining with the structural geopolitical information that flows through the BN.

### 6.3 PyMC implementation sketch

```python
import numpy as np
import pymc as pm

K = 3  # number of regimes
T = len(y_obs)  # number of daily observations

with pm.Model() as model:
    # Prior over regime mixture (could be conditioned on BN parent states)
    pi = pm.Dirichlet("pi", a=np.ones(K) * 2.0)

    # Latent regime assignment for each day
    z = pm.Categorical("z", p=pi, shape=T)

    # Regime-specific emission parameters
    mu = pm.Normal("mu", mu=0.0, sigma=1.0, shape=K)
    sigma = pm.HalfNormal("sigma", sigma=1.0, shape=K)

    # Observed continuous data conditioned on latent regime
    y = pm.Normal("y", mu=mu[z], sigma=sigma[z], observed=y_obs)

    trace = pm.sample()
```

In practice, NUTS cannot sample discrete latent variables. The standard approach is to **marginalize** over $z$ analytically (using `pm.Mixture` or a manual log-likelihood), leaving only continuous parameters for NUTS. This is more efficient and avoids mixing issues.

### 6.4 Coupling to the BN

The BN's posterior over upstream nodes (e.g., the inferred probability of `Strait_Operationally_Closed = full`) can be fed into the PyMC model as an **informative prior** on $\pi$. For example, if the BN posterior says there is 60% chance of strait closure, the Dirichlet concentration for the high-disruption regime should be proportionally larger. This creates a two-layer system:

1. **Layer 1 (discrete BN)**: propagates geopolitical evidence to disruption/scenario posteriors.
2. **Layer 2 (PyMC)**: takes those posteriors as regime priors, adds continuous market data, and produces tighter regime estimates and continuous forecasts.

### 6.5 Extensions beyond Gaussian emissions

For financial returns under stress regimes, heavier-tailed distributions are often more appropriate:

$$
Y_t \mid Z_t = k \sim \text{StudentT}(\nu_k, \mu_k, \sigma_k)
$$

or regime-dependent regressions:

$$
Y_t \mid Z_t = k \sim \mathcal{N}(\beta_{0,k} + \beta_{1,k} X_t, \; \sigma_k)
$$

where $X_t$ could include global risk indicators, inventory surprises, etc.

---

## 7) Extension: richer priors on CPT parameters

### 7.1 The limitation of fixed CPTs

Currently every CPT entry is a fixed constant. The sensitivity analysis (Section 5) probes uncertainty by resampling around these constants, but it does not **learn** — the CPTs do not update as evidence accumulates.

### 7.2 Dirichlet priors over CPT columns

The natural Bayesian extension: treat each CPT column $\theta_j$ as a random variable with a Dirichlet prior:

$$
\theta_j \sim \mathrm{Dirichlet}(\alpha_j)
$$

where $\alpha_j = \kappa_j \cdot m_j$. Here $m_j$ is the current elicited column (the point estimate) and $\kappa_j$ is a scalar controlling confidence:

- $\kappa_j = 5$: weak confidence — a few real observations will substantially shift the column.
- $\kappa_j = 50$: strong confidence — many observations needed to override the prior.

If you later observe $n_j$ actual outcomes for parent configuration $j$, the posterior is:

$$
\theta_j \mid n_j \sim \mathrm{Dirichlet}(\alpha_j + n_j)
$$

This is exact conjugate updating. The current `_resample_cpd` function already uses this Dirichlet structure — the extension is to feed actual observation counts into $n_j$ rather than always sampling around the prior.

### 7.3 Hierarchical priors for related columns

Many CPT columns share qualitative structure (e.g., all "high-escalation" parent combos should push probability toward aggressive child states). A hierarchical model formalizes this:

$$
\eta_j \sim \mathcal{N}(\mu_\eta, \Sigma_\eta), \qquad
\theta_j = \text{softmax}(\eta_j)
$$

where $\mu_\eta$ and $\Sigma_\eta$ are shared hyperparameters estimated across related columns. This **partially pools** information across sparse parent configurations, reducing the risk that a single unusual observation distorts an entire column.

### 7.4 What this buys you

- CPTs that **learn from data** while staying anchored to expert elicitation.
- A principled way to express "I trust this part of the model more than that part."
- Posterior intervals on CPT entries themselves (not just on scenario probabilities), useful for identifying which parts of the model are most uncertain.

---

## 8) Extension: daily updating with partial data and news as observations

### 8.1 Current daily flow (what the app already does)

The dashboard maintains a day counter and an observation log:

1. User advances to day $t$.
2. Headlines arrive and are translated (or nodes are manually set).
3. Each translation appends an `Observation` to the log with day metadata.
4. `_merged_evidence()` builds cumulative evidence (latest-per-node).
5. Inference runs, scenario cards update.

This is already a form of sequential conditioning: $P(\text{Scenario} \mid E_1, E_2, \ldots, E_t)$.

### 8.2 Formalizing the sequential update

Let $D_t$ denote all data (headlines, market closes, manual overrides) arriving on day $t$.

**For the static BN** (current model, no time dynamics), the update at day $t$ is:

$$
P(\text{Scenario} \mid D_{1:t}) = P(\text{Scenario} \mid E_{1:t})
$$

where $E_{1:t}$ is the cumulative evidence set. Each new observation refines the evidence set; inference is re-run from scratch using VE.

**For a dynamic extension** (future, if states can evolve over time), the update becomes:

$$
p(\Theta, Z_{1:t} \mid D_{1:t}) \propto p(D_t \mid Z_t, \Theta) \cdot p(Z_t \mid Z_{t-1}, \Theta) \cdot p(\Theta, Z_{1:t-1} \mid D_{1:t-1})
$$

where $\Theta$ are model parameters (CPT entries or continuous parameters) and $Z_t$ are latent states at time $t$. This would require MCMC or sequential Monte Carlo rather than simple VE.

### 8.3 Hard evidence vs soft evidence

**Hard evidence** (current behavior): a node is clamped to one state with probability 1. Inference conditions on that state exactly.

Example: headline says "US conducts strikes against IRGC naval assets" → translator assigns `US_Military_Response = major`. The posterior treats this as certain.

**Soft evidence** (recommended extension): instead of clamping, provide a **likelihood vector** over all states:

$$
\lambda(\text{node}) = [\lambda_1, \lambda_2, \ldots, \lambda_K]
$$

For example, if a headline *suggests* but does not confirm military action:

$$
\lambda(\text{US\_Military\_Response}) = [0.1, 0.4, 0.5]
$$

This multiplicatively reweights the node's marginal without collapsing it to a point. Soft evidence is particularly useful when:

- the headline is ambiguous
- multiple conflicting sources exist on the same day
- the translator's confidence is moderate

pgmpy supports soft evidence through virtual evidence nodes, though the current codebase does not use this feature.

### 8.4 Incorporating market data as evidence

If continuous market observables are added (Section 6), daily market closes become observations in the PyMC layer. The daily pipeline would:

1. Run the LLM translator on that day's headlines → hard/soft evidence for BN nodes.
2. Run BN inference under the new evidence → updated discrete posteriors.
3. Feed discrete posteriors as priors into the PyMC layer.
4. Condition the PyMC layer on observed market data for that day.
5. Run PyMC inference → updated regime posteriors and continuous forecasts.

### 8.5 Data pipeline for production

1. **Collect**: fetch headlines (RSS, API) and market closes (Brent, freight, CDS) at end of day.
2. **Translate**: run headline batch through translator; collect assignments with confidence scores.
3. **Validate**: check node/state validity; flag contradictions; log warnings.
4. **Store**: append to immutable observation log (extend current `Observation` with confidence and evidence-type fields).
5. **Merge**: build cumulative evidence using latest-per-node policy (or confidence-weighted policy).
6. **Infer**: run BN inference + optional PyMC layer.
7. **Snapshot**: persist posterior probabilities, credible intervals, and observation metadata for audit.
8. **Monitor**: track posterior drift, flag large single-day jumps for review.

### 8.6 Handling contradictions and stale evidence

The current model has no notion of evidence decay. If on day 1 the translator assigns `Tanker_Incidents = frequent` and on day 10 there is no new tanker news, the model still conditions on frequent incidents.

Options for handling this:

- **Explicit retraction**: an analyst or automated check removes stale evidence after $N$ days.
- **Decay weighting**: convert hard evidence to progressively softer evidence over time (likelihood vector moves toward uniform).
- **Time-indexed nodes**: explicitly model `Tanker_Incidents_t` as a time-indexed variable with a transition model, so the model can "forget" old states naturally.

---

## 9) Summary of what is implemented vs proposed

| Feature | Status | Where |
|---------|--------|-------|
| Discrete BN with 13 nodes | Implemented | `src/network.py` |
| Elicited CPTs with validation | Implemented | `src/network.py` |
| Exact inference via VE | Implemented | `src/inference.py` |
| LLM headline → node assignments | Implemented | `src/translator.py` |
| Cumulative evidence, day tracking | Implemented | `app/dashboard.py` |
| Dirichlet-resampled uncertainty bands | Implemented | `src/sensitivity.py` |
| Soft/likelihood evidence | Proposed | Section 8.3 |
| Bayesian learning of CPT parameters | Proposed | Section 7.2 |
| Continuous market observation nodes | Proposed | Section 6 |
| Daily sequential pipeline | Proposed | Section 8.5 |
| Evidence decay / time-indexed nodes | Proposed | Section 8.6 |
