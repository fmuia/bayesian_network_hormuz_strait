"""Elicitation layer embedded in the main dashboard.

A panel that lets you run a real multi-agent elicitation over the whole network
(or load a saved one), read the per-node reasonings and calibration scores the
LLMs produced, override CPT columns, and *lock* the elicitation you like — after
which the rest of the dashboard runs on the locked network (with per-CPT kappa
feeding the credible intervals).

The pure helpers at the top carry the logic and are unit-tested; ``render``
drives the Streamlit widgets.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from src.elicitation.integration import (
    REASONING_EFFORTS,
    EffortConfig,
    ElicitationFramework,
    available_models,
    default_seeds,
    list_runs,
    load_run,
    load_seeds,
    run_elicitation,
    run_to_dict,
    save_run,
    save_seeds,
    slug_id,
)
from src.elicitation.integration.framework import ModelSpec
from src.elicitation.protocols.base import SeedQuestion
from src.network import build_network
from src.network_spec import NetworkSpec

RUNS_DIR = _REPO_ROOT / "data" / "elicitation_runs"
SEEDS_PATH = _REPO_ROOT / "data" / "elicitation_seeds.json"
DEFINITIONAL_NODES = {"Scenario"}  # the terminal node that *defines* the scenarios


def current_seeds() -> list[SeedQuestion]:
    """The analyst's saved seed set, or the illustrative defaults if none saved."""
    return load_seeds(SEEDS_PATH) or default_seeds()


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #


def base_spec(topology: str = "labelling") -> NetworkSpec:
    return NetworkSpec.from_pgmpy(build_network(topology))


def build_framework(
    models: list[ModelSpec],
    n_agents: int,
    nodes: list[str],
    concurrency: int,
    reasoning: str,
    time_limit: float,
    seeds: list[SeedQuestion],
) -> ElicitationFramework:
    return ElicitationFramework(
        name="dashboard",
        models=list(models),
        n_agents=int(n_agents),
        nodes=list(nodes),
        seeds=list(seeds),
        effort=EffortConfig(
            concurrency=int(concurrency),
            reasoning_effort=reasoning,
            time_limit_s=float(time_limit),
        ),
    )


def _enc(config: tuple[str, ...]) -> str:
    return json.dumps(list(config))


def override_in_spec_dict(spec_dict: dict, node: str, config: tuple[str, ...], vector: list[float]) -> dict:
    """Replace one CPT column in a serialised spec (normalised). Returns it."""
    arr = np.clip(np.asarray(vector, dtype=float), 0.0, None)
    if arr.sum() <= 0:
        raise ValueError("override must have positive mass")
    normalised = list(arr / arr.sum())
    for nd in spec_dict["nodes"]:
        if nd["name"] == node:
            key = _enc(tuple(config))
            if key not in nd["cpt"]:
                raise KeyError(f"{node} has no column {config}")
            nd["cpt"][key] = normalised
            return spec_dict
    raise KeyError(f"unknown node {node}")


def locked_spec_json(run_dict: dict) -> str:
    """The spec JSON string the dashboard locks onto."""
    return json.dumps(run_dict["spec"])


def new_run_id() -> str:
    return "run-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


# --------------------------------------------------------------------------- #
# Streamlit rendering
# --------------------------------------------------------------------------- #


def render(st, topology: str = "labelling") -> None:
    st.markdown("#### 🧪 Elicitation layer")
    locked = bool(st.session_state.get("locked_spec_json"))
    if locked:
        st.success("Dashboard is running on a **locked elicited** network.")
        if st.button("↩︎ Revert to bootstrap CPTs"):
            st.session_state["locked_spec_json"] = ""
            st.rerun()
    else:
        st.caption("Dashboard is using the **bootstrap** (hand-authored) CPTs.")

    models = available_models()
    with st.expander("Configure & run", expanded=not locked):
        if not models:
            st.warning(
                "No elicitation models available. Sign in to Claude Code (no key needed) "
                "or set OPENAI_API_KEY for a second model."
            )
        else:
            labels = [m.label for m in models]
            chosen_labels = st.multiselect("Models", labels, default=labels[:1])
            chosen = [m for m in models if m.label in chosen_labels] or models[:1]

            all_nodes = list(base_spec(topology).nodes)
            default_nodes = [n for n in all_nodes if n not in DEFINITIONAL_NODES]
            selected_nodes = st.multiselect(
                "CPTs to elicit", all_nodes, default=default_nodes,
                help="You choose which CPTs to elicit. Unselected ones keep the "
                "hand-authored bootstrap values.",
            )
            st.caption(
                "‘Scenario’ is the definitional terminal node — its CPT *defines* the "
                "three scenarios, so it is left out by default (add it only if you want "
                "the panel to redefine them)."
            )
            c1, c2 = st.columns(2)
            n_agents = c1.number_input("Agents", 1, 8, 3, help="The LLM picks roles per node.")
            reasoning = c2.selectbox(
                "Reasoning budget", list(REASONING_EFFORTS), index=1,
                help="How much each agent deliberates before committing on its first attempt.",
            )
            c3, c4 = st.columns(2)
            concurrency = c3.number_input("Concurrency", 1, 8, 4, help="Parallel calls.")
            time_limit = c4.number_input(
                "Time limit (s)", 10, 300, 45,
                help="Soft budget per call. If an agent runs over, it is re-asked to "
                "conclude now (reusing its reasoning so far) — never abandoned.",
            )

            if not selected_nodes:
                st.warning("Select at least one CPT to elicit.")
            if st.button("▶︎ Run elicitation", type="primary", disabled=not selected_nodes):
                seeds = current_seeds()
                fw = build_framework(chosen, n_agents, selected_nodes, concurrency, reasoning, time_limit, seeds)
                bar = st.progress(0.0, text="starting…")
                run = run_elicitation(
                    base_spec(topology), fw, run_id=new_run_id(),
                    created_at=datetime.now(timezone.utc).isoformat(),
                    on_progress=lambda stage, frac: bar.progress(min(1.0, frac), text=stage),
                )
                save_run(run, RUNS_DIR)
                st.session_state["current_run_dict"] = run_to_dict(run)
                n_boot = len(all_nodes) - len(run.elicited_nodes) - len(run.skipped_nodes)
                summary = (
                    f"Elicited {len(run.elicited_nodes)} CPT(s); "
                    f"{n_boot} kept at bootstrap (not selected). Saved as {run.run_id}."
                )
                if run.skipped_nodes:
                    st.warning(summary + f" ⚠️ FAILED — please re-run: {', '.join(run.skipped_nodes)}.")
                else:
                    st.success(summary)

    _render_seed_editor(st)

    with st.expander("Load a saved run"):
        runs = list_runs(RUNS_DIR)
        if runs:
            label_for = {f"{r['run_id']} · {r['n_elicited']} nodes · {r['models']}": r["run_id"] for r in runs}
            pick = st.selectbox("Saved runs", list(label_for))
            if st.button("Load run"):
                st.session_state["current_run_dict"] = run_to_dict(load_run(label_for[pick], RUNS_DIR))
                st.success(f"Loaded {label_for[pick]}.")
        else:
            st.caption("No saved runs yet.")

    run_dict = st.session_state.get("current_run_dict")
    if run_dict:
        _render_inspector(st, run_dict)


def _render_seed_editor(st) -> None:
    """Edit / add / delete the calibration questions, persisted across sessions."""
    with st.expander("Calibration seeds — add your own questions"):
        st.caption(
            "Known-answer questions used to score each agent. For an LLM, factual "
            "look-up questions mostly test *recall*, not calibration — prefer estimation/"
            "judgment, recent (post-cutoff), counterfactual, or analyst-private figures the "
            "model can't simply look up. Leave the table empty to run equal-weighted (no "
            "calibration)."
        )
        saved = current_seeds()
        rows = [{"question": s.text, "true_value": float(s.realization), "unit": s.unit or ""} for s in saved]
        edited = st.data_editor(
            rows, num_rows="dynamic", width="stretch", key="seed_editor",
            column_config={
                "question": st.column_config.TextColumn("Question", width="large"),
                "true_value": st.column_config.NumberColumn("True value"),
                "unit": st.column_config.TextColumn("Unit"),
            },
        )
        col_a, col_b = st.columns(2)
        if col_a.button("Save seeds"):
            seeds = []
            for i, r in enumerate(edited):
                q = str(r.get("question") or "").strip()
                if not q:
                    continue
                try:
                    val = float(r.get("true_value"))
                except (TypeError, ValueError):
                    continue
                unit = str(r.get("unit") or "").strip() or None
                seeds.append(SeedQuestion(slug_id(q, i), q, val, unit))
            save_seeds(seeds, SEEDS_PATH)
            st.success(f"Saved {len(seeds)} seed(s). They'll be used on the next run.")
        if col_b.button("Reset to illustrative defaults"):
            save_seeds(default_seeds(), SEEDS_PATH)
            st.info("Reset to the illustrative defaults — edit and save to replace them.")


def _render_seeds(st, run_dict: dict) -> None:
    """Show the calibration questions asked, each agent's range, and the scores."""
    seeds = run_dict.get("seeds", [])
    answers = run_dict.get("seed_answers", {})
    agent_names = run_dict["agent_names"]

    if seeds:
        st.markdown("**Calibration questions asked** (each agent gave a 5/50/95 range; ✓ = true value inside it):")
        rows: dict[str, list] = {"Question": [], "True": []}
        for a in agent_names:
            rows[a] = []
        for s in seeds:
            rows["Question"].append(s["text"])
            unit = f" {s['unit']}" if s.get("unit") else ""
            rows["True"].append(f"{s['realization']:g}{unit}")
            for a in agent_names:
                q = answers.get(a, {}).get(s["id"])
                if q:
                    lo, mid, hi = q[0], q[len(q) // 2], q[-1]
                    hit = "✓" if lo <= s["realization"] <= hi else "✗"
                    rows[a].append(f"{mid:g} [{lo:g}–{hi:g}] {hit}")
                else:
                    rows[a].append("—")
        st.dataframe(rows, width="stretch")
    else:
        st.caption("No seed questions recorded for this run.")

    st.markdown("**Per-agent calibration (Cooke seed scores):**")
    for i, sc in enumerate(run_dict["expert_scores"]):
        st.markdown(
            f"{i + 1}. **{sc['name']}** ({sc['model']}) — calibration "
            f"{sc['calibration']:.3f}, weight {sc['weight']:.3f}"
        )
    st.caption(
        "Calibration is the statistical-accuracy p-value from the seeds (higher = better). "
        "The seeds are illustrative — replace with a vetted set for real use."
    )


def _render_override(st, run_dict: dict, node: str, ne: dict) -> None:
    cols = ne["mean_columns"]
    keys = list(cols)

    def _label(k: str) -> str:
        parts = json.loads(k)
        return "prior (no parents)" if not parts else "given " + ", ".join(parts)

    labels = [_label(k) for k in keys]
    st.markdown("**CPT columns** — one distribution per parent combination (a root node has a single *prior* column).")
    choice = st.selectbox("Column", labels, key=f"col_{node}")
    key = keys[labels.index(choice)]
    current = cols[key]
    states = next(nd["states"] for nd in run_dict["spec"]["nodes"] if nd["name"] == node)

    ui_cols = st.columns(len(states))
    new_vals = [
        ui_cols[i].number_input(s, 0.0, 1.0, float(current[i]), 0.01, key=f"ov_{node}_{key}_{s}")
        for i, s in enumerate(states)
    ]
    if st.button("Apply override", key=f"apply_{node}"):
        override_in_spec_dict(run_dict["spec"], node, tuple(json.loads(key)), new_vals)
        ne["mean_columns"][key] = list(np.array(new_vals) / max(sum(new_vals), 1e-9))
        st.session_state["current_run_dict"] = run_dict
        st.success(f"Overrode {node}: {choice}.")


def _render_inspector(st, run_dict: dict) -> None:
    with st.expander("Inspect · override · lock", expanded=True):
        st.caption(run_dict["correlation_note"])

        with st.expander("Seeds & calibration scores", expanded=False):
            _render_seeds(st, run_dict)

        elicited = run_dict["elicited_nodes"]
        if not elicited:
            st.info("No CPTs were elicited in this run.")
        else:
            node = st.selectbox("CPT to inspect / override", elicited)
            ne = run_dict["nodes"][node]
            st.markdown(f"**{node}** — agent-agreement level: **{ne['kappa_level']}**")

            st.markdown("**Panel (one role per agent, chosen by the LLM for this node):**")
            for i, agent in enumerate(run_dict["agent_names"]):
                role = ne["roles"][i] if i < len(ne["roles"]) else "—"
                st.markdown(f"**{i + 1}. {agent}** — _{role}_")
                rationale = ne["rationales"].get(agent, "")
                if rationale:
                    st.caption(rationale)

            st.divider()
            _render_override(st, run_dict, node, ne)

        st.divider()
        if st.button("🔒 Lock this elicitation", type="primary"):
            st.session_state["locked_spec_json"] = locked_spec_json(run_dict)
            st.rerun()
