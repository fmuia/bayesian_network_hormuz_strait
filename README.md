# Strait of Hormuz — Adaptive Scenario Probability Demo

A Bayesian-network proof of concept showing how scenario probabilities for a
Strait of Hormuz crisis shift as news arrives — with a robustified
news→evidence translator and an analyst-in-the-loop review workflow. The
dashboard runs the latent-regime topology (scenarios are latent causes that
generate the observed outcomes), so the headline number is a genuine posterior
`P(Scenario | evidence)`, not a labelling function.

## Why a Bayesian network and not an HMM?

An HMM was considered first but rejected for this layer: the *Severe
Closure* scenario has no historical precedent in market data, so HMM
regime parameters cannot be reliably estimated from price/return series.
A Bayesian network sidesteps this because its conditional probability
tables are elicited from expert reasoning rather than learned from data,
which is exactly what is needed when the scenario of greatest concern is
also the one we have the least data on. The probabilities used here are
illustrative — they are internally consistent with the scenario
narratives, not calibrated against any data set.

## The news → evidence translator

The translator turns a pasted headline (or a full article) into Bayesian-network
evidence. It is the only place natural language enters the model. What it does:

- **Likelihood-ratio semantics.** Each node gets per-state likelihood ratios
  ε ∈ (0, 1] (the best-supported state pinned to 1.0), injected as soft evidence
  — so the BN prior is multiplied exactly once, not double-counted.
- **Article input + source credibility.** Paste a headline, or a full article
  with a source type; less-trusted sources (e.g. state media) are discounted.
- **Relevance / abstention.** Off-topic news is flagged "not relevant" and
  injects nothing, rather than being forced into a mapping.
- **Optional structured pipeline** (sidebar toggle): extract span-grounded
  atomic claims → map each to a node → aggregate. Every assignment cites a
  verbatim span, and instructions embedded in an article ("ignore your prompt…")
  are resisted.
- **Human-in-the-loop Triage** (sidebar toggle): hold translations for
  approve / edit / reject before they affect the model.

### Providers (preference order)

1. **Claude Code** via `claude-agent-sdk` (model `claude-opus-4-8`) — no API key
   needed if the Claude Code CLI is installed and logged in; bills your subscription.
2. **OpenAI** — used if Claude Code is unavailable and `OPENAI_API_KEY` is set.
3. **Fake (offline)** — deterministic fixtures, no network or API key. Force
   with `TRANSLATOR_PROVIDER=fake` or the sidebar "Use fake translator" toggle;
   ideal for demos/dev without spending LLM calls.

If no LLM backend is available you can still set node states by hand in the
Network view. The active provider is shown in the sidebar banner.

## Run

With [pixi](https://pixi.sh):

```
pixi install
pixi run app
```

Or with vanilla pip:

```
pip install -r requirements.txt
streamlit run app/dashboard.py
```

Fully offline (no LLM calls):

```
TRANSLATOR_PROVIDER=fake pixi run app
```

Tests and translator eval:

```
pixi run test             # unit + AppTest suite
pixi run translator-eval  # golden-set contract metrics (offline)
```

The roadmap and what was deliberately deferred for this proof of concept live in
`docs/master_plan.md` and `docs/06_dropped_to_simplify.md`.
