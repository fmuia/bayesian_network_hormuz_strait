# Elicitation Methodology and Defensibility

> **Purpose.** This document is the theory and justification behind the elicitation platform specified in [`docs/04_elicitation_tool_plan.md`](04_elicitation_tool_plan.md). It explains the methods we implement, argues every non-trivial design choice, makes the defensibility case explicit, and states honestly where the approach is contested or rests on a research step. All references in §9 were verified against primary sources (DOI/arXiv/publisher pages); each carries a DOI or canonical URL.
>
> **Reading order.** §1 frames the problem. §2–§3 cover structured expert judgment and the three protocols. §4–§6 cover the quantitative core (aggregation, second-order uncertainty, the calibration→κ mapping). §7 defines the defensible confidence measure. §8 covers AI experts and their specific hazards. §9 is the reference list.

---

## 1. The problem this addresses

The Bayesian network's conditional probability tables (CPTs) are the model's load-bearing assumptions. In the current implementation ([`src/network.py`](../src/network.py)) they are inline literals chosen by one author, with brief comments as justification — no protocol, no multi-expert input, no record of who chose a number, against what reference, or with what confidence, and no tracking of whether a CPT's predictions matched outcomes.

The dashboard already does one thing right: it treats each CPT not as fixed but as the mean of a Dirichlet distribution, resamples all CPTs, re-runs inference, and reports credible intervals ([`src/sensitivity.py`](../src/sensitivity.py)). But it uses a single hard-coded concentration `κ = 20` for every CPT, centred on the one-author means. So the *machinery* for honest uncertainty exists; the *inputs* that would make it meaningful do not. This document sets out the methodology that supplies those inputs: scored multi-expert elicitation for the means, and measured calibration for the per-CPT κ.

The honest ceiling, stated up front: structured expert judgment **manages and measures** the unreliability of expert judgment; it does not remove it. Tetlock's two decades of recorded forecasts found most domain experts barely beat chance, with a trainable minority showing real skill [11, 12]. Everything below is built so that the model finds and weights that minority, and reports its residual uncertainty honestly, rather than trusting expertise on authority.

---

## 2. Structured expert judgment: the field

Structured expert judgment (SEJ) is a mature subfield of decision and risk analysis, with home journals (*Risk Analysis*, *Reliability Engineering & System Safety*, *Decision Analysis*) and institutional backing (EFSA, IPCC, the US Nuclear Regulatory Commission, NICE health-technology appraisals). The standard reference texts are Cooke's *Experts in Uncertainty* [1] and O'Hagan et al.'s *Uncertain Judgements* [8]; the EFSA [9] and IPCC [10] guidance notes are the open institutional standards.

The field is legitimate **and** contested at the same time. The live disagreement is not *whether* to use SEJ but *how to aggregate and validate* it — performance-weighting vs equal-weighting, the dependence on seed questions, and the subjectivity of behavioural consensus. Three protocols span this spectrum; we describe all three below for context, but **only Cooke's classical model is in the platform's implementation scope** (see [plan](04_elicitation_tool_plan.md) §B.6). IDEA and SHELF are documented here and kept behind the same `Protocol`/`Expert` interface so they can be added later without restructuring; nothing builds them now.

---

## 3. The three protocols

### 3.1 SHELF (Sheffield Elicitation Framework)

A facilitator guides a single expert or small group to one consensus ("rational impartial observer") distribution, using roulette (chip-allocation) or quantile elicitation [7]. It is the de facto standard in health-technology assessment. **Known trade-off:** behavioural consensus is facilitator-dependent, vulnerable to groupthink, and yields *no objective accuracy score*. We position SHELF for solo and moderate-stakes work — not as the high-stakes tier.

### 3.2 IDEA

A four-step protocol — **I**nvestigate → individual estimates with credible intervals → **D**iscuss against other experts' estimates → **E**stimate again privately → **A**ggregate [6]. It couples independent estimation (which preserves diversity) with one structured discussion round (which can correct errors). We collect round-1 estimates *before* any cross-expert exposure, and aggregate by linear or geometric pooling. We position IDEA for mid-stakes group work.

### 3.3 Cooke's classical model

The high-stakes tier, and the one this platform leans on. The full workflow:

1. **Define the target** as a clairvoyant-testable quantity. For us, generative emission CPTs under the latent-regime topology ("given the regime, what do the observables look like?").
2. **Build a seed (calibration) question set** — domain questions with known answers, used to measure each expert. Seeds must be *relevant*: they must probe the same judgment as the targets, or measured calibration does not transfer.
3. **Elicit quantiles** from each expert for both seeds and targets, without telling them which are which.
4. **Score each expert** on the seeds, combining two measures:
   - **Calibration (statistical accuracy):** treat the stated quantiles as bins, compare the empirical hit distribution to the asserted one via a likelihood-ratio statistic, and convert to a p-value. Rewards an expert whose stated intervals actually contain the truth at the stated rate. This term dominates the weight.
   - **Information:** entropy of the expert's distributions relative to a background measure. Rewards tight, decisive distributions. Secondary.
5. **Compute weights** as calibration × information, with experts below a significance cutoff **zeroed**. The combined "Decision Maker" is the performance-weighted pool.
6. **Aggregate** the target distributions by the weighted linear pool.
7. **Cross-validate** by leave-one-seed-out, reporting performance-weight vs equal-weight head to head.

Cooke is not fringe: the TU Delft database documents 45 classical-model panels run under contract over ~17 years (>7000 elicited distributions), every one scored against ground truth via seeds [2]; the method produced decision-grade guidance in the Montserrat volcano crisis within hours [3].

### 3.4 Why Cooke is the high-stakes tier — and the live debate

The reason Cooke is the high-stakes tier is precisely that it is the only one of the three that **measures** anything. SHELF and IDEA avoid Cooke's seed-dependence problem only by producing no accuracy score at all — which is not safety, it is silence. For AI panels (§8) this distinction is decisive.

The contested point is real and we do not paper over it: does performance-weighting actually beat equal-weighting *out of sample*? Clemen's cross-validation critique [4] and the Colson & Cooke defence [5] are the two sides. Our response is to make **leave-one-seed-out cross-validation a first-class acceptance test** (Layer 3 validation in the plan) and to report performance-weight vs equal-weight per deployment rather than assume the former wins.

---

## 4. Aggregation primitives

Three pooling rules, each a pure function over expert distributions [13]:

- **Linear pool:** weighted arithmetic mean of the experts' distributions. Cooke prescribes this for the weighted classical-model aggregate.
- **Logarithmic (geometric) pool:** weighted geometric mean, renormalised — more concentrated, externally Bayesian. Offered for IDEA.
- **Cooke (performance-weighted) pool:** a linear pool with the calibration-derived weights of §3.3.

Choice is exposed for IDEA and fixed (linear) for Cooke. The linear-vs-logarithmic question is itself a standing topic in the combining-forecasts literature [13]; we expose it rather than hard-code a position for the protocols where it is genuinely a choice.

---

## 5. Second-order uncertainty: the Dirichlet representation

A CPT column is a categorical distribution $p = (p_1, \dots, p_K)$ with $p_i > 0$, $\sum_i p_i = 1$. We are uncertain about $p$ itself, and we represent that uncertainty with a **Dirichlet** distribution — the natural conjugate over the probability simplex:

$$
f(p \mid \alpha) = \frac{1}{B(\alpha)} \prod_{i=1}^{K} p_i^{\alpha_i - 1}, \qquad \alpha_i > 0 .
$$

The Dirichlet has **one parameter per state** — $K$ parameters for $K$ states, not $2K$. We reparameterise those same $K$ numbers as a **mean vector** plus a single **scalar concentration**:

$$
\kappa \equiv \sum_{i=1}^{K} \alpha_i \quad(\text{scalar}), \qquad m_i \equiv \frac{\alpha_i}{\kappa} \quad\Big(\textstyle\sum_i m_i = 1\Big), \qquad \alpha_i = \kappa\, m_i .
$$

So $m$ carries the $K-1$ free "where" numbers and $\kappa$ carries one "how sure" number. $\kappa$ is the **equivalent pseudo-sample size**: a Dirichlet with concentration $\kappa$ behaves like having seen $\kappa$ prior observations. The per-component variance is

$$
\operatorname{Var}(p_i) = \frac{m_i(1 - m_i)}{\kappa + 1} .
$$

Large $\kappa$ ⇒ tight around the mean (confident); $\kappa \to \infty$ ⇒ point mass (no uncertainty); small $\kappa$ ⇒ diffuse. This is exactly the representation the current dashboard uses with $\alpha = \kappa \cdot m$ and the hard-coded $\kappa = 20$; the methodology below replaces that one global constant with a per-CPT, calibration-derived value.

**Propagation to a credible interval.** Each elicited CPT becomes a Dirichlet. To get the uncertainty on a model output (e.g. $P(S = \text{crisis} \mid \text{evidence})$) we Monte-Carlo propagate: draw one concrete probability vector from every CPT's Dirichlet (yielding one fully-specified network), run ordinary inference, repeat $M$ times, and read the credible interval off the percentiles of the resulting posterior samples. The width of that interval is the model's parametric (epistemic) uncertainty, and it is finite precisely because the $\kappa$'s are finite. If every $\kappa \to \infty$ the interval collapses to a point. This is the mechanism by which **panel quality propagates into final confidence**.

This second-order-probability ("probability of probabilities") treatment is standard practice in the elicitation literature [8] and is the conjugate-Bayesian way to carry epistemic uncertainty about a categorical parameter.

---

## 6. The calibration→κ mapping

This is the hinge of the whole approach, and the one genuinely novel, not-yet-standard step (closest published precedent: AutoElicit's LLM→Gaussian-prior result for linear models [19], which we extend to categorical CPT columns / Dirichlet priors). We set $\kappa$ from measured quality, by two complementary routes, then discretise for reporting.

### 6.1 Route A — coverage-fit on seeds

Because seeds have known answers, we can choose $\kappa$ so the panel's predictive intervals have **correct empirical coverage**: the $\kappa$ for which, across the seed set, the truth falls inside the panel's $X\%$ predictive interval about $X\%$ of the time. Too-narrow intervals (truth escapes too often) ⇒ lower $\kappa$; too-wide ⇒ raise $\kappa$. The fitted $\kappa$ is then applied to the targets. This ties $\kappa$ to demonstrated calibration rather than a guess. (Coverage calibration is natural for continuous seeds; the categorical-CPT analogue, fit via a proper scoring rule, is the open research step noted in the plan's open questions.)

### 6.2 Route B — method-of-moments from panel disagreement

A multi-model AI panel gives a second-order estimate almost for free. Given $N$ agent point vectors $\{\theta^{(n)}\}$, take the sample mean $m$ and per-component sample variance $s_i^2$, and invert the variance formula:

$$
\hat\kappa_i = \frac{m_i(1 - m_i)}{s_i^2} - 1 ,
$$

pooling the $\hat\kappa_i$ across components into a single $\kappa$. The panel's disagreement *is* a proxy for epistemic uncertainty.

**Correlation discount.** Agents off the same base model are not independent, so their apparent diversity overstates the independent information. We replace $N$ by an effective sample size $N_{\text{eff}} = N / (1 + (N-1)\bar\rho)$, where $\bar\rho$ is the measured mean inter-agent correlation, and shrink $\kappa$ accordingly (widen the interval). This is why an honest AI panel produces *wider*, not falsely tighter, intervals — the correlation we measure pushes the confidence down. The empirical basis for taking correlation seriously is the silicon-crowd result: a *diverse* multi-model ensemble matched a human crowd, whereas a *single* model did not [20, 21].

### 6.3 The three-level ordinal ladder, snap-and-gate

LLMs (and humans) are unreliable at fine-grained numeric self-calibration but more robust at coarse ordinal distinctions, and a continuous $\kappa$ advertises a precision the source cannot support. So we **report** $\kappa$ on a three-level ladder — `tight` / `normal` / `uncertain` — with per-deployment fitted values, and:

- **Estimate** $\kappa$ via Route A and/or B (rigorously, continuously).
- **Snap** the estimate to the nearest level for provenance and the dashboard (honest about the real resolution).
- **Gate**: an expert's measured seed-calibration **caps** the level it may contribute — a poorly-calibrated agent cannot claim `tight`.
- **Cross-check**: an agent's self-reported level is logged and compared to the empirical one; a divergence (self-says-`tight`, empirics say `uncertain`) is surfaced as a flag.

Indicatively, on a $p \approx 0.5$ cell, `tight` (κ≈40), `normal` (κ≈15), and `uncertain` (κ≈5) imply roughly ±10 / ±16 / ±26 percentage-point bands — which align with the dashboard's existing 🟢/🟡/🔴 robustness tiers, so the κ ladder and the badge language reinforce rather than duplicate each other. The three ordinal levels also map cleanly onto the IPCC two-dimensional confidence language [10], which is the reporting standard we adopt (§7).

This combines the robustness and honesty of an ordinal scale with the defensibility of a calibration-anchored estimate. The failure mode it avoids is the degenerate version where the level is just an agent's introspective confidence with no seed check behind it.

---

## 7. The defensible confidence measure

"How confident are we in the final output?" has a defensible answer, but it is a **small vector, not a scalar** — and the defensibility comes precisely from refusing to collapse it. A single clean "confidence = 87%" for a model like this is not defensible.

First, a distinction that is constantly conflated: the model's posterior (e.g. $P(\text{crisis} \mid E) = 0.70$) is the **first-order** point estimate; it is *not* the confidence. Confidence is the **second-order** question of how reliable that 0.70 is. We report four components:

1. **Point posterior** — the model's best estimate.
2. **Propagated credible interval** (§5) — the parametric uncertainty from the per-CPT κ's, correlation-discounted for AI panels. Labelled **conditional on the model structure**.
3. **Variance decomposition** — a Sobol-style attribution [14, 15, 16] of the interval width to individual CPTs, so a reviewer sees where the fragility lives and where tightening κ would pay off (Morris screening [14] for cheap pre-screening; Sobol indices [15] for the quantitative attribution; both via SALib [17]).
4. **Empirical calibration track record** — reliability diagrams and Brier scores from realised outcomes (Tier 2/3). This is the *only* out-of-sample component and the only one that can catch a **wrong structure**, which the interval in (2) cannot self-detect.

These are wrapped in an **IPCC-style two-dimensional confidence statement** [10]: a likelihood (the calibrated probability) reported separately from a confidence rating (evidence quality × agreement), with an explicit structural-uncertainty caveat. We adopt the IPCC scheme because it is purpose-built for exactly this — high-stakes judgment under deep uncertainty — and is defensible by institutional adoption.

Two honest limits, stated in every report: the interval captures parametric uncertainty **conditional on structure** (only the empirical track record can catch a wrong DAG), and until outcome data accrues the confidence is **model-internal** — an estimate of reliability, not demonstrated reliability.

---

## 8. AI experts: the case, the hazards, the mitigations

### 8.1 Positioning

We position this as **calibration-validated AI elicitation, not "AI replaces experts."** The enabler is that Cooke's seed scoring is itself a calibration test: an LLM expert answers the same seeds a human would, is scored identically, and is weighted — or zeroed — on that evidence. This converts "should we trust the AI's numbers?" from faith into a measured, auditable quantity. The operating mode is AI-only panels with mandatory human sign-off for high-stakes work; the `Expert` interface is human-capable so human panellists are a configuration choice.

The premise that an LLM's probabilities are measurable and sometimes usable is supported: larger models are reasonably calibrated on well-formatted multiple-choice/true-false and can partly self-predict what they know [18]; models can emit calibrated verbalized confidence [22]; and a retrieval-augmented system can approach the human-crowd aggregate on genuinely post-cutoff questions [23]. The "mostly" in [18] is the catch — which is why calibration is measured on *domain* seeds, not generic benchmarks.

### 8.2 The correlation problem

Agents off the same base model share training data and biases, so naive pooling overstates confidence. The empirical backbone is decisive: a **diverse multi-model ensemble** matched a 925-person human crowd [20], whereas a **single model** was no better than chance [21] — diversity is doing the work. Mitigations, all recorded in provenance: compose panels from genuinely different base models (not personas/temperatures of one); estimate and report inter-agent correlation; and shrink the effective sample size and κ (§6.2). Multi-agent debate can improve factuality [24], but whether it adds information or merely amplifies the shared prior is exactly the correlation risk — so we measure post-discussion calibration against round-1 rather than assume debate helps.

**Roles and personas — useful, but not a source of independence.** Assigning each agent a role or character (a red-team skeptic, a base-rate thinker, an escalation-pessimist) is a legitimate way to widen the considerations a panel surfaces and to drive structured debate. But it must not be confused with diversity: five personas on *one* base model remain one correlated source, because they share the same weights and training data. Roles therefore **compose with** base-model diversity and are **excluded from the independence count** in the effective-sample-size calculation (§6.2). Two consequences for calibration. First, the calibration unit is the **`(base_model, role, config)` tuple**, not the model — a role-conditioned estimate is a different estimator, so an agent seed-scored "neutral" carries no valid calibration when it then answers "in role." Second, a persona that deliberately biases an estimate (a hawk talking up escalation) is acceptable as a scored contributor *only if it is seed-scored in that same persona*; otherwise the role belongs to a divergence/brainstorm pass that surfaces considerations without directly setting the scored probability.

### 8.3 The contamination problem — and why "post-cutoff seeds" is not a free lunch

An AI's seed score is inflated if it *recalls* a memorised answer rather than *reasoning*. Naively "use post-cutoff seeds" is partly self-deception: training cutoffs are fuzzy and undisclosed; a niche domain is seed-starved; calibration is perishable across model upgrades; and contamination is unprovable (you cannot prove the model never saw something), made worse by RAG retrieving the answer into context. Even contamination-free, a seed score measures calibration on *resolved, lookup-able-in-principle* quantities, whereas the real targets are latent regimes and counterfactual dynamics that may never resolve cleanly — an irreducible transfer gap.

The defensible posture is therefore **defense in depth**, with seed calibration treated as a **filter that removes poorly-calibrated agents, never a certificate that licenses trust**:

- **Prospective scoring is primary.** The only contamination-proof seed is one whose answer does not exist yet. We score agents (and humans) on genuinely unresolved future domain events, logged immutably with timestamps and scored on resolution (the Tier 2/3 substrate). This is contamination-proof by construction; its cost is latency.
- **Retrodictive post-cutoff seeds are a flagged bootstrap** for a provisional day-one weight — explicitly low-confidence until prospective data accrues, and the practice of rolling, post-cutoff, leakage-limited benchmarks is the field's response to static-benchmark rot [25, 30].
- **Active contamination probes** rather than assumed absence: source-attribution (ask where it knows the answer — if it can cite the source, discard the seed), perturbation/canary (change a date/entity; if confidence does not move when the answer should, it is recall) [29], in-corpus-vs-post-cutoff split scoring to quantify inflation, and anomalously-low cross-model variance as a leakage alarm.
- **Relevance constraint.** Seeds must probe the same judgment as the targets, or calibration does not transfer.

### 8.4 Knowledge currency

An LLM's "expertise" is its training corpus — public and frozen at the cutoff. It does not natively know proprietary material or this-week intelligence; RAG over the translator audit log supplies that, but then the agent is *summarising retrieved evidence*, not exercising privileged judgment. We treat AI experts as strong on broad, public, slow-moving structure and weak on fresh or proprietary specifics, and let seed-calibration on domain questions — not credentials — set their weight.

### 8.5 Sycophancy, false consensus, and judge independence

LLMs tend toward agreement regardless of truth (sycophancy) [26], so the discussion steps (IDEA's discuss-and-revise, SHELF's facilitated consensus) can manufacture confident, unanimous error. Mitigations: collect round-1 estimates before any cross-exposure; assign at least one adversarial/red-team agent prompted to refute; weight by calibrated confidence rather than forcing consensus; and measure whether discussion improves calibration against round-1. Where an agent *judges* another agent, the judge must be a **different base model** — LLM judges exhibit position, verbosity, and self-preference biases [27, 28], so self-grading is prohibited and answer order is randomised. Self-consistency (sampling multiple reasoning paths and taking the modal answer) [31] is available for an individual agent's own outputs.

### 8.6 The research contribution and its honest framing

A focused search did not surface a standardised, published method that marries Cooke's classical model (seed-validated, performance-weighted, poor experts zeroed) to an *LLM* expert panel feeding a Bayesian network's CPTs. The components are validated separately — structured expert judgment [1–13], LLM forecasting/calibration [18–23], multi-agent aggregation and LLM-elicited priors [24, 19]. Assembling them is both a methodological advance and a genuine research contribution, to be validated with the rigour the SEJ literature demands: domain seed sets, leave-one-seed-out cross-validation, explicit correlation accounting, and the Tetlock ceiling [11, 12] kept in view. The deliverable is *calibrated, measured, auditable* judgment — not *reliable* judgment.

### 8.7 The defensibility requirement, concretely

Every AI-sourced CPT is flagged in `cpt_provenance` with: its calibration score, its κ level, the model set, the inter-agent correlation note, and a contamination-probe summary — plus the provenance of the calibration itself (which seeds, their resolution dates vs the model's stated cutoff, retrodictive-vs-prospective split). Human sign-off is mandatory for high-stakes work. This is what makes the approach auditable to a regulator: not "we made the AI reliable," but "we measured, diversified, adversarially checked, and logged everything, and here is exactly how much we trust it and why."

---

## 9. References

All entries verified against primary sources (DOI / arXiv / publisher pages). Books carry an ISBN where no DOI exists; preprints carry the arXiv ID and the peer-reviewed venue where one exists.

### 9.1 Structured expert judgment and elicitation methodology

1. Cooke, R. M. (1991). *Experts in Uncertainty: Opinion and Subjective Probability in Science*. New York: Oxford University Press. ISBN 9780195064650.
2. Cooke, R. M., & Goossens, L. L. H. J. (2008). TU Delft expert judgment data base. *Reliability Engineering & System Safety*, 93(5), 657–674. doi:10.1016/j.ress.2007.03.005
3. Aspinall, W. (2010). A route to more tractable expert advice. *Nature*, 463(7279), 294–295. doi:10.1038/463294a
4. Clemen, R. T. (2008). Comment on Cooke's classical method. *Reliability Engineering & System Safety*, 93(5), 760–765. doi:10.1016/j.ress.2008.02.003
5. Colson, A. R., & Cooke, R. M. (2018). Expert elicitation: Using the classical model to validate experts' judgments. *Review of Environmental Economics and Policy*, 12(1), 113–132. doi:10.1093/reep/rex022
6. Hemming, V., Burgman, M. A., Hanea, A. M., McBride, M. F., & Wintle, B. C. (2018). A practical guide to structured expert elicitation using the IDEA protocol. *Methods in Ecology and Evolution*, 9(1), 169–180. doi:10.1111/2041-210X.12857
7. Gosling, J. P. (2018). SHELF: The Sheffield Elicitation Framework. In L. C. Dias, A. Morton, & J. Quigley (Eds.), *Elicitation: The Science and Art of Structuring Judgement* (International Series in Operations Research & Management Science, Vol. 261, pp. 61–93). Cham: Springer. doi:10.1007/978-3-319-65052-4_4
8. O'Hagan, A., Buck, C. E., Daneshkhah, A., Eiser, J. R., Garthwaite, P. H., Jenkinson, D. J., Oakley, J. E., & Rakow, T. (2006). *Uncertain Judgements: Eliciting Experts' Probabilities*. Chichester: Wiley. doi:10.1002/0470033312
9. EFSA (European Food Safety Authority) (2014). Guidance on Expert Knowledge Elicitation in Food and Feed Safety Risk Assessment. *EFSA Journal*, 12(6), 3734. doi:10.2903/j.efsa.2014.3734
10. Mastrandrea, M. D., Field, C. B., Stocker, T. F., Edenhofer, O., Ebi, K. L., Frame, D. J., Held, H., Kriegler, E., Mach, K. J., Matschoss, P. R., Plattner, G.-K., Yohe, G. W., & Zwiers, F. W. (2010). *Guidance Note for Lead Authors of the IPCC Fifth Assessment Report on Consistent Treatment of Uncertainties*. IPCC. https://www.ipcc.ch/site/assets/uploads/2018/05/uncertainty-guidance-note.pdf
11. Tetlock, P. E. (2005). *Expert Political Judgment: How Good Is It? How Can We Know?* Princeton, NJ: Princeton University Press. ISBN 9780691128719.
12. Tetlock, P. E., & Gardner, D. (2015). *Superforecasting: The Art and Science of Prediction*. New York: Crown. ISBN 9780804136693.
13. Clemen, R. T., & Winkler, R. L. (1999). Combining probability distributions from experts in risk analysis. *Risk Analysis*, 19(2), 187–203. doi:10.1111/j.1539-6924.1999.tb00399.x

### 9.2 Sensitivity analysis

14. Morris, M. D. (1991). Factorial sampling plans for preliminary computational experiments. *Technometrics*, 33(2), 161–174. doi:10.1080/00401706.1991.10484804
15. Sobol', I. M. (2001). Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. *Mathematics and Computers in Simulation*, 55(1–3), 271–280. doi:10.1016/S0378-4754(00)00270-6
16. Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J., Gatelli, D., Saisana, M., & Tarantola, S. (2008). *Global Sensitivity Analysis: The Primer*. Chichester: Wiley. doi:10.1002/9780470725184
17. Herman, J., & Usher, W. (2017). SALib: An open-source Python library for sensitivity analysis. *Journal of Open Source Software*, 2(9), 97. doi:10.21105/joss.00097

### 9.3 LLM calibration and forecasting

18. Kadavath, S., Conerly, T., Askell, A., et al. (2022). *Language Models (Mostly) Know What They Know*. arXiv:2207.05221. https://arxiv.org/abs/2207.05221 (Anthropic technical report; preprint.)
19. Capstick, A., Krishnan, R. G., & Barnaghi, P. (2024). *AutoElicit: Using Large Language Models for Expert Prior Elicitation in Predictive Modelling*. arXiv:2411.17284. ICML 2025. https://arxiv.org/abs/2411.17284
20. Schoenegger, P., Tuminauskaite, I., Park, P. S., Valdece Sousa Bastos, R., & Tetlock, P. E. (2024). Wisdom of the silicon crowd: LLM ensemble prediction capabilities rival human crowd accuracy. *Science Advances*, 10(45), eadp1528. doi:10.1126/sciadv.adp1528. arXiv:2402.19379
21. Schoenegger, P., & Park, P. S. (2023). *Large Language Model Prediction Capabilities: Evidence from a Real-World Forecasting Tournament*. arXiv:2310.13014. https://arxiv.org/abs/2310.13014 (Preprint.)
22. Lin, S., Hilton, J., & Evans, O. (2022). *Teaching Models to Express Their Uncertainty in Words*. Transactions on Machine Learning Research (TMLR). arXiv:2205.14334. https://arxiv.org/abs/2205.14334
23. Halawi, D., Zhang, F., Chen Yueh-Han, & Steinhardt, J. (2024). *Approaching Human-Level Forecasting with Language Models*. NeurIPS 2024. arXiv:2402.18563. https://arxiv.org/abs/2402.18563

### 9.4 Multi-agent LLMs, evaluation reliability, and contamination

24. Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). *Improving Factuality and Reasoning in Language Models through Multiagent Debate*. ICML 2024 (PMLR 235:11733–11763). arXiv:2305.14325. https://arxiv.org/abs/2305.14325
25. Karger, E., Bastani, H. (Houtan Bastani), Chen Yueh-Han, Jacobs, Z., Halawi, D., Zhang, F., & Tetlock, P. E. (2024). *ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities*. ICLR 2025. arXiv:2409.19839. https://arxiv.org/abs/2409.19839
26. Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S. R., et al. (2023). *Towards Understanding Sycophancy in Language Models*. ICLR 2024. arXiv:2310.13548. https://arxiv.org/abs/2310.13548
27. Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023 (Datasets & Benchmarks Track). arXiv:2306.05685. https://arxiv.org/abs/2306.05685
28. Panickssery, A., Bowman, S. R., & Feng, S. (2024). *LLM Evaluators Recognize and Favor Their Own Generations*. NeurIPS 2024. arXiv:2404.13076. https://arxiv.org/abs/2404.13076
29. Golchin, S., & Surdeanu, M. (2023). *Time Travel in LLMs: Tracing Data Contamination in Large Language Models*. ICLR 2024. arXiv:2308.08493. https://arxiv.org/abs/2308.08493
30. White, C., Dooley, S., Roberts, M., Pal, A., Feuer, B., et al. (2024). *LiveBench: A Challenging, Contamination-Limited LLM Benchmark*. ICLR 2025. arXiv:2406.19314. https://arxiv.org/abs/2406.19314
31. Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2022). *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. ICLR 2023. arXiv:2203.11171. https://arxiv.org/abs/2203.11171

---

**Companion document:** [`docs/04_elicitation_tool_plan.md`](04_elicitation_tool_plan.md) — the executable layered plan this methodology justifies.
