"""Sensitivity analysis.

Two general-purpose screeners over a user-supplied model ``f`` on a box domain
(implemented directly in numpy/scipy rather than via SALib to keep the
dependency surface small), and the domain-specific
:func:`posterior_variance_decomposition` that attributes a Bayesian-network
output's uncertainty to individual CPTs — the prioritiser Layer 6 consumes.

* :func:`morris_screening` — elementary-effects screening: ``mu_star`` (mean
  absolute effect) and ``sigma`` per input. Cheap, qualitative.
* :func:`sobol_indices` — first-order and total Sobol indices via the
  Saltelli/Jansen estimators. Quantitative.
* :func:`posterior_variance_decomposition` — per-CPT share of output variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

ModelFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def _scale(unit: NDArray[np.float64], bounds: Sequence[tuple[float, float]]) -> NDArray[np.float64]:
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return lo + unit * (hi - lo)


@dataclass(frozen=True)
class MorrisResult:
    mu_star: NDArray[np.float64]
    sigma: NDArray[np.float64]


def morris_screening(
    f: ModelFn,
    bounds: Sequence[tuple[float, float]],
    trajectories: int = 200,
    delta: float = 0.1,
    seed: int | None = 0,
) -> MorrisResult:
    """Elementary-effects screening.

    For each trajectory a random base point is perturbed one input at a time;
    the elementary effect is the normalised output change. Returns ``mu_star``
    (mean absolute effect — overall influence) and ``sigma`` (its spread —
    non-linearity / interactions) per input.
    """
    rng = np.random.default_rng(seed)
    d = len(bounds)
    effects: list[list[float]] = [[] for _ in range(d)]
    for _ in range(trajectories):
        base_unit = rng.uniform(0, 1 - delta, size=d)
        base = _scale(base_unit[None, :], bounds)
        y0 = float(np.asarray(f(base)).ravel()[0])
        for i in range(d):
            pert_unit = base_unit.copy()
            pert_unit[i] += delta
            y1 = float(np.asarray(f(_scale(pert_unit[None, :], bounds))).ravel()[0])
            effects[i].append((y1 - y0) / delta)
    eff = np.array(effects)  # (d, trajectories)
    return MorrisResult(mu_star=np.mean(np.abs(eff), axis=1), sigma=np.std(eff, axis=1))


@dataclass(frozen=True)
class SobolResult:
    first_order: NDArray[np.float64]
    total: NDArray[np.float64]


def sobol_indices(
    f: ModelFn,
    bounds: Sequence[tuple[float, float]],
    n: int = 4096,
    seed: int | None = 0,
) -> SobolResult:
    """First-order and total Sobol indices via the Saltelli/Jansen estimators.

    ``f`` must be vectorised: it takes an ``(m, d)`` array and returns ``(m,)``.
    """
    rng = np.random.default_rng(seed)
    d = len(bounds)
    a_unit = rng.uniform(size=(n, d))
    b_unit = rng.uniform(size=(n, d))
    fa = np.asarray(f(_scale(a_unit, bounds)), dtype=float).ravel()
    fb = np.asarray(f(_scale(b_unit, bounds)), dtype=float).ravel()

    var = np.var(np.concatenate([fa, fb]))
    first = np.zeros(d)
    total = np.zeros(d)
    for i in range(d):
        ab_unit = a_unit.copy()
        ab_unit[:, i] = b_unit[:, i]
        fab = np.asarray(f(_scale(ab_unit, bounds)), dtype=float).ravel()
        # Saltelli (2010) first-order; Jansen total.
        first[i] = np.mean(fb * (fab - fa)) / var
        total[i] = 0.5 * np.mean((fa - fab) ** 2) / var
    return SobolResult(first_order=first, total=total)


@dataclass(frozen=True)
class VarianceDecomposition:
    total_variance: float
    contributions: dict[str, float]   # raw main-effect variance per CPT
    shares: dict[str, float]          # normalised to sum to 1


def posterior_variance_decomposition(
    cpts: Mapping[str, tuple[Sequence[float], float]],
    output_fn: Callable[[Mapping[str, NDArray[np.float64]]], float],
    n_samples: int = 512,
    seed: int | None = 0,
) -> VarianceDecomposition:
    """Attribute a scalar BN output's variance to individual CPTs.

    ``cpts`` maps a CPT id to ``(mean_vector, kappa)``. ``output_fn`` maps a fully
    sampled set of CPT vectors to a scalar (e.g. ``P(crisis | evidence)``). The
    main-effect contribution of each CPT is the output variance when only that
    CPT is resampled and the rest are held at their means; shares are normalised.
    This is the "where would tightening kappa most reduce the interval" map.
    """
    rng = np.random.default_rng(seed)
    means = {k: np.asarray(m, dtype=float) for k, (m, _) in cpts.items()}
    kappas = {k: float(kap) for k, (_, kap) in cpts.items()}

    def draw(node: str) -> NDArray[np.float64]:
        return rng.dirichlet(kappas[node] * means[node])

    total_outputs = np.array(
        [output_fn({k: draw(k) for k in cpts}) for _ in range(n_samples)]
    )
    total_variance = float(np.var(total_outputs))

    contributions: dict[str, float] = {}
    for target in cpts:
        outputs = np.array(
            [
                output_fn({k: (draw(k) if k == target else means[k]) for k in cpts})
                for _ in range(n_samples)
            ]
        )
        contributions[target] = float(np.var(outputs))

    denom = sum(contributions.values())
    shares = (
        {k: v / denom for k, v in contributions.items()}
        if denom > 0
        else {k: 0.0 for k in contributions}
    )
    return VarianceDecomposition(
        total_variance=total_variance, contributions=contributions, shares=shares
    )
