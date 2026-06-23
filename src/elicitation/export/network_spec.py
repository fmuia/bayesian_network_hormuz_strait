"""Convert between the elicitation store and a declarative ``NetworkSpec``.

``cpts_to_network_spec`` builds a spec from a deployment's stored CPTs (with
their per-CPT kappa); ``network_spec_to_cpts`` imports a spec (e.g. the bootstrap
pack network) as a refinement starting point. The node structure (states,
parents) is stored inside the versioned ``cpt_versions.values`` JSON blob, so no
schema change is needed and snapshots are self-describing.
"""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from ...network_spec import CptTable, DiscreteNode, NetworkSpec
from ..db.schema import Cpt, CptVersion, Network


def _encode_key(key: tuple[str, ...]) -> str:
    return json.dumps(list(key))


def _decode_key(text: str) -> tuple[str, ...]:
    return tuple(json.loads(text))


def _encode_values(node: DiscreteNode) -> dict:
    return {
        "states": list(node.states),
        "parents": list(node.parents),
        "columns": {_encode_key(k): list(v) for k, v in node.cpt.items()},
    }


def _decode_values(blob: dict) -> tuple[tuple[str, ...], tuple[str, ...], CptTable]:
    states = tuple(blob["states"])
    parents = tuple(blob["parents"])
    cpt = {_decode_key(k): list(v) for k, v in blob["columns"].items()}
    return states, parents, cpt


def network_spec_to_cpts(
    session: Session,
    spec: NetworkSpec,
    network_name: str,
    topology: str = "latent_regime",
) -> int:
    """Persist a NetworkSpec into the elicitation store. Returns the network id."""
    spec.validate()
    net = Network(name=network_name, topology=topology)
    session.add(net)
    session.flush()
    for node in spec.nodes.values():
        cpt = Cpt(network_id=net.id, node=node.name)
        session.add(cpt)
        session.flush()
        version = CptVersion(
            cpt_id=cpt.id,
            version=1,
            values=_encode_values(node),
            kappa=node.kappa,
            kappa_level=node.kappa_level,
        )
        session.add(version)
        session.flush()
        cpt.current_version_id = version.id
    session.commit()
    return int(net.id)


def cpts_to_network_spec(
    session: Session,
    network_id: int,
    snapshot_at: datetime | None = None,
) -> NetworkSpec:
    """Build a NetworkSpec from a deployment's stored CPTs.

    Uses each CPT's current version, or — if ``snapshot_at`` is given — the
    latest version in force at that time.
    """
    spec = NetworkSpec()
    cpts = session.scalars(select(Cpt).where(Cpt.network_id == network_id)).all()
    for cpt in cpts:
        version = _version_in_force(session, cpt, snapshot_at)
        if version is None:
            continue
        states, parents, table = _decode_values(version.values)
        spec.add(
            DiscreteNode(
                name=cpt.node,
                states=states,
                parents=parents,
                cpt=table,
                kappa=version.kappa,
                kappa_level=version.kappa_level,
            )
        )
    return spec


def _version_in_force(
    session: Session, cpt: Cpt, snapshot_at: datetime | None
) -> CptVersion | None:
    if snapshot_at is None:
        if cpt.current_version_id is None:
            return None
        return session.get(CptVersion, cpt.current_version_id)
    stmt = (
        select(CptVersion)
        .where(CptVersion.cpt_id == cpt.id, CptVersion.created_at <= snapshot_at)
        .order_by(CptVersion.version.desc())
    )
    return session.scalars(stmt).first()


# -- JSON (de)serialisation for the CLI -------------------------------------- #


def spec_to_dict(spec: NetworkSpec) -> dict:
    return {
        "nodes": [
            {
                "name": n.name,
                "states": list(n.states),
                "parents": list(n.parents),
                "cpt": {_encode_key(k): list(v) for k, v in n.cpt.items()},
                "kappa": n.kappa,
                "kappa_level": n.kappa_level,
            }
            for n in spec.nodes.values()
        ]
    }


def spec_from_dict(data: dict) -> NetworkSpec:
    spec = NetworkSpec()
    for nd in data["nodes"]:
        spec.add(
            DiscreteNode(
                name=nd["name"],
                states=tuple(nd["states"]),
                parents=tuple(nd["parents"]),
                cpt={_decode_key(k): list(v) for k, v in nd["cpt"].items()},
                kappa=nd.get("kappa"),
                kappa_level=nd.get("kappa_level"),
            )
        )
    return spec


__all__ = [
    "network_spec_to_cpts",
    "cpts_to_network_spec",
    "spec_to_dict",
    "spec_from_dict",
]
