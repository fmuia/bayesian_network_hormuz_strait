"""Persist elicitation runs as JSON artifacts (the 'series of saved ones').

A saved run holds the resulting network (means + per-CPT kappa), the per-node
reasonings and scores, and the panel metadata — everything the dashboard needs
to load a run, inspect it, override columns, and lock it.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..export import spec_from_dict, spec_to_dict
from .runner import ElicitationRun, NodeElicitation


def _enc(cfg: tuple[str, ...]) -> str:
    return json.dumps(list(cfg))


def _dec(text: str) -> tuple[str, ...]:
    return tuple(json.loads(text))


def _node_to_dict(ne: NodeElicitation) -> dict:
    return {
        "node": ne.node,
        "roles": ne.roles,
        "rationales": ne.rationales,
        "mean_columns": {_enc(k): v for k, v in ne.mean_columns.items()},
        "per_expert": {
            agent: {_enc(k): v for k, v in cols.items()} for agent, cols in ne.per_expert.items()
        },
        "kappa": ne.kappa,
        "kappa_level": ne.kappa_level,
        "error": ne.error,
    }


def _node_from_dict(d: dict) -> NodeElicitation:
    return NodeElicitation(
        node=d["node"],
        roles=d["roles"],
        rationales=d["rationales"],
        mean_columns={_dec(k): v for k, v in d["mean_columns"].items()},
        per_expert={a: {_dec(k): v for k, v in cols.items()} for a, cols in d["per_expert"].items()},
        kappa=d["kappa"],
        kappa_level=d["kappa_level"],
        error=d.get("error"),
    )


def run_to_dict(run: ElicitationRun) -> dict:
    return {
        "run_id": run.run_id,
        "created_at": run.created_at,
        "framework_name": run.framework_name,
        "n_agents": run.n_agents,
        "agent_names": run.agent_names,
        "models": run.models,
        "expert_scores": run.expert_scores,
        "elicited_nodes": run.elicited_nodes,
        "skipped_nodes": run.skipped_nodes,
        "correlation_note": run.correlation_note,
        "seeds": run.seeds,
        "seed_answers": run.seed_answers,
        "spec": spec_to_dict(run.spec),
        "nodes": {name: _node_to_dict(ne) for name, ne in run.nodes.items()},
        "diagnostics": run.diagnostics,
        "contamination": run.contamination,
    }


def run_from_dict(data: dict) -> ElicitationRun:
    return ElicitationRun(
        run_id=data["run_id"],
        created_at=data["created_at"],
        framework_name=data["framework_name"],
        n_agents=data["n_agents"],
        agent_names=data["agent_names"],
        models=data["models"],
        expert_scores=data["expert_scores"],
        elicited_nodes=data["elicited_nodes"],
        skipped_nodes=data["skipped_nodes"],
        correlation_note=data["correlation_note"],
        seeds=data.get("seeds", []),
        seed_answers=data.get("seed_answers", {}),
        spec=spec_from_dict(data["spec"]),
        nodes={name: _node_from_dict(nd) for name, nd in data["nodes"].items()},
        diagnostics=data.get("diagnostics"),
        contamination=data.get("contamination"),
    )


def save_run(run: ElicitationRun, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run.run_id}.json"
    path.write_text(json.dumps(run_to_dict(run), indent=2))
    return path


def load_run(run_id: str, directory: str | Path) -> ElicitationRun:
    return run_from_dict(json.loads((Path(directory) / f"{run_id}.json").read_text()))


def list_runs(directory: str | Path) -> list[dict]:
    """Return run metadata (id, created_at, framework, counts) for each saved run."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("*.json")):
        d = json.loads(path.read_text())
        out.append(
            {
                "run_id": d["run_id"],
                "created_at": d["created_at"],
                "framework_name": d["framework_name"],
                "models": d["models"],
                "n_elicited": len(d["elicited_nodes"]),
                "n_skipped": len(d["skipped_nodes"]),
            }
        )
    return out


__all__ = ["run_to_dict", "run_from_dict", "save_run", "load_run", "list_runs"]
