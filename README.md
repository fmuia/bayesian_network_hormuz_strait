# Strait of Hormuz — Adaptive Scenario Probability Demo

A toy Bayesian-network demo showing how scenario probabilities for a Strait
of Hormuz crisis shift as new evidence arrives.

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

Tests:

```
pixi run test
```
