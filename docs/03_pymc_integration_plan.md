# PyMC Integration and Continuous-Variable Migration Plan

> **Status.** Draft. No phases started.
>
> **Related docs.** `docs/master_plan.md` §4 is the in-tree registry of finding IDs and lists the findings this plan closes (M2, M3, M4). M1 and M7 (latent regime) used to be in this plan as Phase 3; they have been consolidated into `docs/01_latent_regime_plan.md` (Plan 1) so the conceptual decision and the engineering implementation live together. This plan still provides the `PymcBackend` substrate Plan 1's engineering depends on (Phase 2 here is the gating dependency), and Plan 1's engineering must land before this plan's continuous-variable phases (Phases 3–4) can build on it. `docs/bn_hmm_integration.md` describes the longer-horizon BN↔HMM story, **out of scope for this plan** — it requires a separately-trained inflation HMM that does not exist in this repo.
>
> **Status legend.** ⬜ not started · ⏳ in progress · ✅ shipped (with date).

## Executive Summary

This document is the plan to migrate the inference backend from pgmpy to PyMC, with the migration designed so that the existing pgmpy backend can continue to serve the all-discrete case at lower cost. The migration unlocks three capabilities the current architecture cannot deliver cleanly:

1. **Hierarchical priors over CPTs.** Resolves M2 (soft evidence semantics), M3 (uniform κ), and M4 (independent CPT resampling) in a single architectural move — they share a common shape that PyMC supports natively. M1 and M7 (latent regime) are addressed by Plan 1, which builds on this plan's `PymcBackend` substrate.
2. **Continuous variables.** Several nodes in the current model (Oil_Price_Regime, Conflict_Duration, Energy_Infrastructure_Damage) are forced discretizations of inherently continuous quantities.
3. **Modern Bayesian diagnostics.** Posterior diagnostics, R-hat, ESS, divergence tracking — the PPL ecosystem standard.

The migration is structured as a **dual-backend system**. A declarative `NetworkSpec` describes the model; a dispatcher routes to either `PgmpyBackend` (when the network is all-discrete and the user prefers it) or `PymcBackend` (when continuous variables are present, or when the user opts in for richer features). The dashboard layer stays backend-agnostic.

The plan is five phases (0 through 4). Phases 0–2 are pure refactoring with no semantic change — they deliver the `NetworkSpec`, the `PgmpyBackend` wrapper, and `PymcBackend` for discrete networks. **Plan 1's engineering implementation slots between Phase 2 and Phase 3** (Plan 1 promotes the latent regime to a first-class topology on top of `PymcBackend`'s discrete substrate). Phases 3–4 then build on top of Plan 1's latent regime: continuous-variable support (Phase 3) and the production Oil_Price migration (Phase 4). Temporal extensions and BN↔HMM integration are out of scope; see `docs/bn_hmm_integration.md` for the longer-horizon story.

## Section A — Motivation and the math

### A.1 Why this migration

Two converging motivations make a backend change worth the engineering cost:

**1. Hierarchical CPT priors.** The M-series findings in the dashboard review (M2 soft-evidence semantics, M3 uniform $\kappa$, M4 independent CPT column resampling) all share a common shape: they want CPT entries themselves to be random variables with explicit prior distributions, not point values with bolt-on Dirichlet resampling. The current `src/sensitivity.py` is a post-hoc workaround. Under PyMC, the Dirichlet prior *is* the CPT prior, and parameter uncertainty becomes inference, not perturbation. M1 and M7 (the latent-regime findings) are owned by Plan 1, which builds on the `PymcBackend` substrate this plan delivers.

**2. Continuous variables.** Three nodes in the current model are "secretly continuous" and have been forced into 3-state discretizations: Oil_Price_Regime ($\in \{$below_90, 90_to_120, above_120$\}$), Conflict_Duration ($\in \{$short, medium, long$\}$), Energy_Infrastructure_Damage ($\in \{$none, moderate, severe$\}$). All three lose information in the discretization. PyMC handles mixed discrete/continuous models natively; pgmpy does not.

Temporal extensions to the BN (a Markov chain on the `Scenario` variable) and BN↔HMM integration are out of scope for this plan — they require a separately-trained inflation HMM that does not exist in this repo and are tracked in `docs/bn_hmm_integration.md` as a longer-horizon workstream. Roadmap item A1 (evidence accumulation) therefore remains unaddressed by this plan; see `docs/master_plan.md` §6 (Gaps).

### A.2 What stays mathematically the same

Five foundational properties survive the migration unchanged:

- **The DAG structure.** Same nodes, same edges, same factorization story.
- **The chain rule of BNs.** $P(X_1, \ldots, X_n) = \prod_i P(X_i \mid \text{Pa}(X_i))$.
- **Conditional independence and d-separation.** Graphical rules don't depend on whether nodes are discrete or continuous.
- **Bayes' rule.** Posterior is proportional to prior times likelihood, regardless of variable types.
- **The latent-regime framework.** $S$ remains a latent categorical; its parents and children just become a richer set of distribution types.

The migration is about *how the joint is represented and how inference is performed*, not about the conceptual model.

### A.3 What changes mathematically

The seven specific mathematical changes that drive the engineering work.

#### Change 1 — CPTs become families of conditional distributions

In the discrete world, a CPT for node $Y$ with parents $\text{Pa}(Y)$ is a table:

$$
P(Y = y \mid \text{Pa}(Y) = u) = \theta_{y \mid u}, \quad \sum_y \theta_{y \mid u} = 1
$$

In the mixed world, the conditional distribution of $Y$ given $\text{Pa}(Y)$ becomes a **parametric family** whose parameters depend on the parent values:

$$
Y \mid \text{Pa}(Y) = u \;\sim\; F\big(\boldsymbol{\eta}(u)\big)
$$

where $F$ is a distribution family (Gaussian, LogNormal, Gamma, Beta) and $\boldsymbol{\eta}(u)$ is a parameter vector that may depend on $u$. Three cases that come up in practice:

- **Continuous child of discrete parents.** Oil_Price $\mid$ (Strait, Damage) $\sim$ LogNormal$(\mu_{c,d}, \sigma_{c,d})$ — a $3 \times 3 = 9$-cell parameter table.
- **Continuous child of continuous parents.** Oil_Price $\mid$ Duration $\sim$ LogNormal$(\mu_0 + \beta \log(1+t), \sigma)$ — a parametric regression.
- **Discrete child of continuous parents.** Conflict_Continues $\mid$ Oil_Price $\sim$ Bernoulli$(\sigma(\alpha + \beta p))$ — logistic regression in place of a CPT row.

The general pattern: **conditional distributions are parameterized by their parents via small functional forms**. Elicitation moves from "fill in 27 cells" to "choose a family and pick a few coefficients." Vastly more compact, more interpolation-friendly, but a different kind of elicitation effort.

#### Change 2 — The joint mixes pmfs and pdfs

In the discrete world the joint $P(X_1, \ldots, X_n)$ is a function on a finite product space that sums to 1. In the mixed world the joint is a **mixed-type density**:

$$
\pi(x_1, \ldots, x_n) = \prod_i \pi_i(x_i \mid \text{pa}_i)
$$

where each $\pi_i$ is a pmf if $X_i$ is discrete, a pdf if continuous. Normalization mixes summation and integration:

$$
\sum_{x \in \text{disc}} \int_{x \in \text{cont}} \pi(\boldsymbol{x}) \, dx_{\text{cont}} \;=\; 1
$$

#### Change 3 — Inference goes from exact summation to numerical integration

The fundamental inference operation is structurally the same:

$$
\pi(X_i \mid E) = \frac{\pi(X_i, E)}{\pi(E)}
$$

But the integral / sum over unobserved variables in mixed models generally has no closed form. Three numerical strategies:

- **Exact VE on conjugate sub-models.** Works for purely-Gaussian sub-networks (Lauritzen-Jensen). Breaks for non-Gaussian or nonlinear continuous nodes.
- **MCMC.** Sample $\theta^{(1)}, \ldots, \theta^{(N)}$ from the joint posterior using NUTS (No-U-Turn Sampler) or its variants. Estimate expectations via

$$
\mathbb{E}[f(X) \mid E] \approx \frac{1}{N} \sum_{n=1}^N f(\theta^{(n)})
$$

Asymptotically exact; standard error $\sim 1/\sqrt{N \cdot \text{ESS}}$.

- **Variational inference (VI).** Approximate the posterior with a tractable parametric family $q_\phi$, minimize $\text{KL}(q_\phi \| \pi(\cdot \mid E))$. Faster than MCMC but produces biased posteriors. PyMC supports ADVI.

For interactive dashboard use, MCMC is the right default — asymptotically exact, with controllable error bars. For low-cardinality discrete latents (like the regime $S$), analytic marginalization is preferred over sampling.

#### Change 4 — Continuous evidence has richer types

Discrete evidence: "node $X = $ state $s$." That's it.

Continuous evidence supports:

- **Point observation.** $X = 142.3$ (delta likelihood).
- **Interval observation.** $X \in [140, 150]$ (integral of density over interval).
- **Noisy observation.** Observed $Y \mid X = x \sim \mathcal{N}(x, \sigma^2_{\text{obs}})$ (measurement-error sub-model).
- **Censored observation.** "$X$ exceeded threshold at some point" (truncated likelihood).

Each of these requires a corresponding term in the likelihood, expressed via PyMC's `pm.Potential` mechanism.

#### Change 5 — Densities vs probabilities (communication implications)

$P(X = 0.94)$ for a discrete state is a probability. $p(X = 0.94)$ for a continuous variable is a density and can exceed 1. Stakeholder-facing language shifts:

- **Discrete result:** "Oil_Price = above_120 has probability 0.45."
- **Continuous result:** "Oil_Price has posterior median 148, with 80% CI [122, 184], and $P(\text{Oil_Price} > 120) = 0.78$."

The decision-relevant statement for continuous nodes is the **probability of being in a range**, computed by integration. The dashboard layer needs to support interval queries (`P(X > τ | E)`) and density plots, in addition to discrete state probability bars.

#### Change 6 — Hierarchical uncertainty becomes one inference pass

Currently there are two layers of uncertainty machinery:

- **Inference uncertainty.** Variable elimination on point-estimate CPTs.
- **Parameter uncertainty.** Dirichlet resampling perturbs CPTs and reruns inference $m=200$ times.

Under PyMC with hierarchical priors, the two collapse into one:

$$
\pi(S, \boldsymbol{\theta}_\text{CPT} \mid E) \;\propto\; \pi(S \mid \boldsymbol{\theta}_\text{CPT}) \cdot \pi(\boldsymbol{\theta}_\text{CPT}) \cdot \mathcal{L}(E \mid S, \boldsymbol{\theta}_\text{CPT})
$$

MCMC samples the joint posterior. The regime marginal is obtained by ignoring all columns except $S$ in the trace:

$$
\pi(S \mid E) = \int \pi(S, \boldsymbol{\theta}_\text{CPT} \mid E) \, d\boldsymbol{\theta}_\text{CPT}
$$

— captured automatically by the sampling. No bolt-on resampling needed.

#### Change 7 — Soft evidence becomes a proper likelihood

The M2 mismatch (translator produces a posterior-shaped distribution, pgmpy treats it as a likelihood) goes away because PyMC requires you to write the likelihood explicitly:

```python
# Choice 1: treat translator output as likelihood directly
log_lik = pt.log(q_translator[X])
pm.Potential("translator_obs", log_lik)

# Choice 2: convert posterior-shaped output to likelihood by dividing by prior
log_lik = pt.log(q_translator[X]) - pt.log(p_marginal_prior[X])
pm.Potential("translator_obs", log_lik)
```

Either way, the semantic claim is now an explicit and auditable function rather than an implicit consequence of library convention. M2 is resolved by *commitment to one of these forms*, not by library magic.

### A.4 Summary table of mathematical changes

| Aspect | Discrete-only (current) | Mixed (PyMC) |
| --- | --- | --- |
| Conditional distributions | Tables | Parametric families parameterized by parents |
| Joint distribution | Product of pmfs | Mixed product of pmfs and pdfs |
| Inference | Exact variable elimination | MCMC (NUTS + CompoundStep, or VI) |
| Uncertainty | Two layers (point + Dirichlet resampling) | One joint posterior over (latents, CPT params) |
| Evidence | Categorical assignments only | Points, intervals, noisy, censored |
| Likelihood | Implicit in virtual-evidence convention | Explicit via `pm.Potential` |
| Elicitation | Fill in CPT cells | Choose distribution family + functional form |
| Output | Probabilities (sums to 1) | Densities + integrals + samples |
| Tractability | Polynomial in clique size | Asymptotically exact MCMC; cost in sampling time |
| Output uncertainty | CIs via post-hoc resampling | CIs from joint posterior samples natively |

## Section B — Architecture

### B.1 Design principle: dual backend with capability-based dispatch

The migration is structured so that:

- **If the network has continuous variables → PyMC is the only viable backend.**
- **If the network is all-discrete → user can choose pgmpy (faster, exact) or PyMC (richer features).**

A declarative `NetworkSpec` describes the network; a dispatcher routes to the appropriate backend; both backends produce a uniform `Posterior` object that the dashboard consumes. This means the dashboard layer never knows or cares which backend computed its inputs.

```
┌──────────────────────────────────────┐
│ NetworkSpec (declarative)            │
│ - nodes: List[NodeSpec]              │
│ - edges: List[Tuple[str, str]]       │
│ - CPTs / emission distributions       │
│ - per-CPT priors (Dirichlet κ etc.)   │
└──────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│ Backend dispatcher                   │
│ - has_continuous_node(spec)?         │
│ - user preference (auto/pgmpy/pymc)  │
└──────────────────────────────────────┘
        │                       │
        ▼                       ▼
┌──────────────┐         ┌──────────────┐
│ PgmpyBackend │         │ PymcBackend  │
│ (discrete    │         │ (any         │
│  only, fast) │         │  structure)  │
└──────────────┘         └──────────────┘
        │                       │
        └───────────┬───────────┘
                    ▼
        ┌─────────────────────┐
        │ Posterior (uniform  │
        │  shape across       │
        │  backends)          │
        └─────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │ Dashboard           │
        └─────────────────────┘
```

### B.2 The `NetworkSpec` data structure

Pure data, no backend logic. Sketch:

```python
# src/network_spec.py
from dataclasses import dataclass
from typing import Callable, Literal, Union
import numpy as np

@dataclass
class DiscreteNode:
    name: str
    states: list[str]
    parents: list[str]
    cpt: np.ndarray              # shape (state_dim, *parent_dims)
    kappa: float = 20.0          # Dirichlet concentration for parameter uncertainty
    provenance: str = ""

@dataclass
class ContinuousNode:
    name: str
    parents: list[str]
    distribution: Literal["normal", "lognormal", "gamma", "beta"]
    parameter_fn: Callable        # (parent_values) -> distribution parameters
    kappa_params: dict[str, float]   # hyperprior concentration per parameter
    provenance: str = ""

NodeSpec = Union[DiscreteNode, ContinuousNode]

@dataclass
class NetworkSpec:
    nodes: list[NodeSpec]

    def has_continuous(self) -> bool:
        return any(isinstance(n, ContinuousNode) for n in self.nodes)

    def topological_order(self) -> list[str]: ...
    def validate_dag(self) -> None: ...
```

### B.3 The backend interface

```python
# src/backends/base.py
from abc import ABC, abstractmethod

class InferenceBackend(ABC):
    @classmethod
    @abstractmethod
    def supports_continuous(cls) -> bool: ...

    @abstractmethod
    def __init__(self, spec: NetworkSpec): ...

    @abstractmethod
    def query(
        self,
        evidence: dict[str, Any],
        soft_evidence: dict[str, dict] = None,
        target_nodes: list[str] = None,
    ) -> Posterior: ...

    @abstractmethod
    def credible_intervals(
        self,
        evidence: dict[str, Any],
        soft_evidence: dict[str, dict] = None,
        m: int = 200,
        ci: float = 0.80,
    ) -> CIDict: ...
```

### B.4 The dispatcher

```python
# src/backends/__init__.py
def make_backend(
    spec: NetworkSpec,
    prefer: Literal["auto", "pgmpy", "pymc"] = "auto",
) -> InferenceBackend:
    has_continuous = spec.has_continuous()

    if has_continuous:
        if prefer == "pgmpy":
            raise ValueError(
                "pgmpy backend doesn't support continuous variables. "
                "Use prefer='auto' or prefer='pymc'."
            )
        return PymcBackend(spec)

    # All-discrete case: user can choose
    match prefer:
        case "pgmpy":
            return PgmpyBackend(spec)
        case "pymc":
            return PymcBackend(spec)
        case "auto":
            return PgmpyBackend(spec)  # default to pgmpy for discrete (faster, exact)
```

### B.5 The uniform `Posterior` object

```python
# src/posterior.py
@dataclass
class Posterior:
    """Backend-agnostic posterior representation."""

    _marginals: dict[str, MarginalDist]
    _samples: dict[str, np.ndarray] | None      # available for PymcBackend
    _provenance: dict[str, str]                  # which backend, sampling params

    def marginal(self, node: str) -> MarginalDist: ...
    def credible_interval(self, node: str, ci: float = 0.80) -> tuple[float, float]: ...
    def mean(self, node: str) -> float | np.ndarray: ...
    def samples(self, node: str) -> np.ndarray | None: ...   # None for pgmpy
    def bayes_factor(self, ...) -> float: ...                # only meaningful for latent regime
    def probability_of_interval(self, node: str, lo: float, hi: float) -> float: ...
```

Continuous nodes expose `probability_of_interval` and density-based queries. Discrete nodes expose the standard categorical marginal API.

## Section C — Phased plan

Each phase has a clear scope, deliverable, and validation criterion. Phases 0–2 are pure refactoring with no semantic change. Phases 3–5 are progressive scope expansion. Temporal extensions and BN↔HMM coupling are out of scope; see `docs/bn_hmm_integration.md`.

### Phase 0 — Refactor to introduce `NetworkSpec`

**Status.** ⬜ not started

**Scope.** Extract the existing `src/network.py` into a `NetworkSpec` data structure. No backend code yet — just the declarative representation. Existing tests continue to pass via a thin wrapper that converts `NetworkSpec` to a pgmpy `DiscreteBayesianNetwork`.

**Deliverables.**

- `src/network_spec.py` — `NetworkSpec`, `DiscreteNode`, `ContinuousNode` (placeholder), validation.
- `src/network.py` — rewritten as a function `build_hormuz_spec() -> NetworkSpec`.
- Helper: `_spec_to_pgmpy(spec) -> DiscreteBayesianNetwork` so existing code keeps working.

**Validation.**

- All existing tests pass without modification.
- New test verifies `build_hormuz_spec().has_continuous() == False`.
- New test verifies `_spec_to_pgmpy(build_hormuz_spec())` produces a network identical to the current `build_network()`.

### Phase 1 — Introduce `PgmpyBackend`

**Status.** ⬜ not started

**Scope.** Wrap the existing inference machinery (`BNInferenceEngine`, `sensitivity.py`) in the `InferenceBackend` interface. Build the uniform `Posterior` object as a thin wrapper around pgmpy outputs. Dashboard updated to consume `Posterior` instead of raw pgmpy outputs.

**Deliverables.**

- `src/backends/base.py` — abstract `InferenceBackend` and supporting types.
- `src/backends/pgmpy_backend.py` — concrete implementation.
- `src/posterior.py` — uniform output type.
- `app/dashboard.py` — refactored to consume `Posterior`.

**Validation.**

- All existing dashboard behavior preserved (same numbers in scenario cards, same CIs).
- Tests verify dispatcher: `make_backend(build_hormuz_spec())` returns `PgmpyBackend`.
- Tests verify dispatcher honors `prefer="pgmpy"` and rejects continuous specs under `prefer="pgmpy"`.

### Phase 2 — Build `PymcBackend` for discrete-only networks

**Status.** ⬜ not started

**Scope.** Implement `PymcBackend` that handles purely-discrete networks. Translates `NetworkSpec` → PyMC model with Dirichlet priors on CPTs and Categorical nodes. Sampling via NUTS + CompoundStep for discrete categoricals, or analytic marginalization for small-cardinality latents. Produces uniform `Posterior` output.

**Deliverables.**

- `src/backends/pymc_backend.py` — concrete implementation.
- `src/backends/_pymc_helpers.py` — categorical CPT construction, analytic marginalization helpers, evidence injection (hard + soft via `pm.Potential`).

**Validation.**

- Posterior summaries from `PgmpyBackend` and `PymcBackend` agree to within MCMC error on a battery of test evidence configurations. Specifically:
  - Prior scenario distribution: match within 0.005.
  - Single-evidence updates: match within 0.01.
  - Compound-evidence updates: match within 0.02.
- Sampling is seeded and reproducible.
- Diagnostics: R-hat < 1.01 across all parameters; ESS > 400 for scenario marginals.

> **Note.** Plan 1 (`docs/01_latent_regime_plan.md`) owns the latent-regime conceptual decision and engineering implementation, including the resolution of findings M1 and M7. **Plan 1's engineering implementation slots between Phase 2 and Phase 3 below**: Phase 2 ships the `PymcBackend` substrate Plan 1 builds on; Phase 3 (continuous variables) depends on Plan 1's latent regime being in place because the continuous emissions are emissions *of the latent regime*.

### Phase 3 — Continuous variable support in `PymcBackend`

**Status.** ⬜ not started
**Depends on.** Plan 1's engineering implementation (latent regime) having landed.
**Resolves.** Review findings M2 (soft evidence as proper likelihood), M3 (per-CPT $\kappa$ — the `NetworkSpec`-side mechanism; the elicitation-side mechanism that populates κ is Plan 4 Layer 4), and M4 (independent CPT column resampling — under PyMC's hierarchical priors, all rows of a CPT share a single Dirichlet parameter draw per posterior sample, so correlated shape uncertainty is the natural output of one inference pass rather than a post-hoc resampling artefact).

**Scope.** Extend `NetworkSpec` to support `ContinuousNode`. `PymcBackend` translates these to PyMC continuous distributions (LogNormal, Gamma, Beta) with regime-dependent parameters. Dispatcher routes any spec with continuous nodes to `PymcBackend` unconditionally.

Posterior object gains continuous-variable APIs (`probability_of_interval`, `density`, `quantile`). Dashboard gains rendering for continuous posteriors (density plots, interval queries).

**Deliverables.**

- `src/network_spec.py` — `ContinuousNode` activated.
- `src/backends/pymc_backend.py` — handles continuous nodes.
- `src/posterior.py` — continuous-variable query API.
- Continuous-variable UI components live in Plan 5 (`app/components/continuous_viz.py` — density plot, interval query). Phase 3 here ships only the backend hooks; the rendering work is owned by Plan 5 Category C (item C14).

**Validation.**

- A test spec with one continuous node (a continuous Oil_Price replacing the 3-state version) produces sensible posteriors.
- Hard observations on the continuous node update the regime posterior in expected directions.
- Soft observations (interval, noisy) work and produce sensible updates.

### Phase 4 — Migrate Oil_Price to continuous in production

**Status.** ⬜ not started
**Depends on.** Phase 3 above (continuous-variable support in `PymcBackend`).

**Scope.** Promote `Oil_Price_Regime` from discrete 3-state to continuous LogNormal. Update emission CPT, translator (to handle continuous observations: point, interval, noisy), and dashboard visualization for the continuous node.

Oil-price model:

$$
\text{OilPrice} \mid S, C, D \sim \text{LogNormal}\!\big(\mu(S, C, D), \sigma(S, C, D)\big)
$$

with regime-dependent mean and variance. Initial parameter values anchored to historical analog events (1990 Gulf War, 1979 oil shock, 2008 supply concerns).

**Deliverables.**

- `src/cpt_data.py` — continuous emission parameters for Oil_Price.
- `src/translator.py` — extended to recognize continuous observations from headlines.
- Dashboard rendering (density plot for Oil_Price posterior, interval-probability queries on the scenario cards: $P(\text{Oil} > 120)$ etc.) is owned by Plan 5 Category C item C14. Phase 4 here ships the backend data; Plan 5 ships the surfaces.

**Validation.**

- Real oil-price observations (from a market data source) feed into the model and shift the regime posterior in expected directions.
- The continuous Oil_Price posterior has reasonable shape (positive skew, regime-dependent location and scale).
- Stakeholder-facing language updated to use interval probabilities: "78% probability oil exceeds $120."

## Section D — Decision points and exit options

Each phase has a natural exit point if you want to stop:

- **After Phase 1.** Backend abstraction in place; can keep pgmpy as the only backend. Value: cleaner architecture, no semantic change.
- **After Phase 2.** PyMC available as opt-in for the existing discrete model. Validated parity between backends. Plan 1's engineering implementation can begin from this point.
- **After Plan 1's engineering** (between this plan's Phase 2 and Phase 3). Latent regime in production; M1 and M7 resolved. Bayes factors as first-class outputs.
- **After Phase 3.** Continuous variables possible. M2 / M3 / M4 also naturally resolved.
- **After Phase 4.** One real continuous node (Oil_Price) in production.

Temporal extensions and BN↔HMM integration sit beyond Phase 4 and are tracked in `docs/bn_hmm_integration.md`; they require a separately-trained inflation HMM and are out of scope for this plan.

## Section E — Open questions

These do not block Phase 0 but should be resolved before the corresponding phase begins:

| Question | Block | Notes |
| --- | --- | --- |
| Sampler choice for discrete latents in Phase 2 | Phase 2 | Default: analytic marginalization for $S$ (low cardinality, exact). NUTS + CompoundStep for larger discrete sets. Decide if needed beyond 3-state. |
| Per-CPT $\kappa$ values for the latent-regime emission CPTs | Plan 1 (resolved: provisional in Plan 1, elicited in Plan 4 Layer 4) | Plan 1 ships uniform $\kappa = 20$ on the anchor-derived emission CPTs; Plan 4 Layer 4 round-trips elicited per-CPT $\kappa$ values into `cpt_provenance` and `PymcBackend` consumes them from there. See M3 in the review for the underlying motivation. |
| Continuous oil-price source data | Phase 4 | Bloomberg, FRED, Quandl, EIA? Decide data source and update cadence. |
| Translator extension for continuous observations | Phase 4 | Headlines like "oil hit $148" — extract as point observations. "Oil between $140-150 this week" — extract as interval. LLM prompt extension needed. |

## Section F — Execution order summary table

For coherence with the format used in `docs/bn_app_next_steps.md`:

| Order | Phase | Resolves | Rationale |
| --- | --- | --- | --- |
| 1 | 0 — NetworkSpec refactor | Architecture | Foundational. Pure refactor, no semantic change. Unblocks every subsequent phase. |
| 2 | 1 — PgmpyBackend | Architecture | Wrap existing inference behind backend interface. Dashboard becomes backend-agnostic. |
| 3 | 2 — PymcBackend (discrete) | Architecture | Validated dual-backend support. Discrete-only opt-in for PyMC. |
| — | (Plan 1 engineering slots in here) | M1, M7 | Latent regime built on top of `PymcBackend`. See `docs/01_latent_regime_plan.md` Section B. |
| 4 | 3 — Continuous variable support | M2 (soft evidence semantics), M3 (per-CPT κ), M4 (correlated CPT shape uncertainty) | Hierarchical priors over CPTs; continuous nodes possible; all-rows-share-one-Dirichlet-draw closes M4. Depends on Plan 1's engineering being in place. |
| 5 | 4 — Migrate Oil_Price to continuous | Operational | First production continuous node. Interval-probability queries on the dashboard. |

---

**End of plan.** Companion plan with latent-regime math and engineering: `docs/01_latent_regime_plan.md`. M-series findings registry: `docs/master_plan.md` §4.
