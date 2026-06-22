"""Model-explanation prose: the "How this model works" overview and the
appendix (Plan 5 P4d). Extracted from the dashboard."""
from __future__ import annotations

from src.scenario import PRESENTATION


def render_overview(st, topology) -> None:
    """Render the active pack's model-overview copy for this topology."""
    overview = PRESENTATION.model_overview.get(topology) or next(
        iter(PRESENTATION.model_overview.values()), ""
    )
    if overview:
        st.markdown(overview, unsafe_allow_html=True)


def render_appendix(st) -> None:
    st.markdown(
        r"""
        ### Appendix: implementation details

        The model is a discrete Bayesian network with posterior updates via exact variable elimination.

        **Inference rule**

        $$
        P(S \mid E=e) =
        \frac{\sum_{z} P(S, z, e)}{\sum_{s}\sum_{z} P(s, z, e)}
        $$

        where $S$ is the scenario node and $z$ are latent/unobserved nodes.

        **Translator-to-evidence pipeline**

        1. A headline is parsed into a set of node assignments constrained to valid node states.
        2. Each assignment is appended as an observation with day and source metadata.
        3. Latest assignment wins on node conflicts when merged into current evidence.
        4. Inference is re-run and scenario cards + node marginals are refreshed.

        **Uncertainty panel**

        Credible intervals are estimated by resampling every CPT column from a Dirichlet distribution centred on the elicited point estimate and rerunning inference:

        $$
        \theta_{j,\cdot}^{(m)} \sim \text{Dirichlet}(\alpha_{j,\cdot}),
        \qquad \alpha_{j,\cdot} = \kappa_j \cdot \theta_{j,\cdot}^{\text{point}}
        $$

        with a **per-CPT** concentration $\kappa_j$ — calibrated from elicitation quality when an elicitation is locked (so more-trusted CPTs are perturbed less), otherwise the bootstrap default $\kappa_j = 20$ — and $m = 200$ draws. Each draw perturbs **all** CPTs jointly and the full network is re-run under the current evidence, so the resulting posterior samples reflect *global* parameter uncertainty, not a per-node local variation.

        The 10th–90th percentiles across samples give an 80% credible interval per node per state, exposed in two places:

        - **Scenario cards** (top band): headline CIs for the three scenarios.
        - **Node detail panel** (right of the Network tab): per-node dumbbells with a robustness badge (🟢 robust < ±8 pp · 🟡 moderate ±8–20 pp · 🔴 fragile > ±20 pp). Hard-observed nodes collapse to deltas; soft-observed nodes keep their CIs because the posterior still varies under CPT resampling.
        """
    )


# ---------------------------------------------------------------------------
# Translator stream (compact single-line)
# ---------------------------------------------------------------------------
