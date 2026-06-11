"""Model-explanation prose: the "How this model works" overview and the
appendix (Plan 5 P4d). Extracted from the dashboard."""
from __future__ import annotations

from src.network import SCENARIO_NARRATIVES
from theme import AMBER, GREEN, RED


def render_overview(st, topology) -> None:
    if topology == "latent_regime":
        scenario_clause = (
            "a latent <b>Scenario</b> regime that <i>generates</i> the damage, "
            "duration, and diplomatic-path outcomes (with context parents US "
            "military response and strait closure)"
        )
    else:
        scenario_clause = (
            "a terminal <b>Scenario</b> node classified from the damage, "
            "duration, and diplomatic-path outcomes"
        )
    st.markdown(
        "<div class='explain'>"
        "<p>The Bayesian network encodes qualitative causal structure "
        "between four <b>root drivers</b> (negotiations, regime "
        "stability, third-party mediation, sanctions trajectory), "
        "<b>eight intermediate nodes</b> (Iran-aligned militia attacks, tanker "
        "incidents, US military response, strait closure, energy "
        "infrastructure damage, conflict duration, diplomatic path, "
        f"oil price regime), and {scenario_clause} "
        "whose three states correspond to the client's strategic "
        "scenarios.</p>"
        "<h4>Two layers</h4>"
        "<p>A free-text headline is passed through an LLM translator "
        "that extracts BN-relevant probabilistic assignments (e.g. "
        "<i>\"Fourth tanker incident in two weeks\"</i> gives a high "
        "probability to <code>Tanker_Incidents = frequent</code>). "
        "Those soft assignments become BN evidence; variable-elimination "
        "propagates them and yields the posterior distribution at "
        "every node.</p>"
        "<h4>Scenario definitions</h4>"
        "<ul>"
        f"<li><b style='color:{GREEN};'>Stress Mitigates</b> — "
        f"{SCENARIO_NARRATIVES['Stress_Mitigates']}</li>"
        f"<li><b style='color:{AMBER};'>Prolonged Conflict</b> — "
        f"{SCENARIO_NARRATIVES['Prolonged_Conflict']}</li>"
        f"<li><b style='color:{RED};'>Severe Closure</b> — "
        f"{SCENARIO_NARRATIVES['Severe_Closure']}</li>"
        "</ul>"
        "<h4>Reading the graph</h4>"
        "<p>Teal-filled nodes are the ones for which evidence has "
        "been set (whether by the translator or a manual override). "
        "Unobserved nodes display the most likely state under the "
        "current posterior. Root drivers use distinct color families "
        "so they are easy to distinguish visually.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


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
