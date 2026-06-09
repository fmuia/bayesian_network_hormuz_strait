"""Tests for the Layer 0 database schema and per-deployment isolation."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect

from src.elicitation.db import create_all, make_engine, make_session_factory
from src.elicitation.db.schema import (
    Cpt,
    CptProvenance,
    CptVersion,
    Network,
    ProviderCredentialRow,
)

EXPECTED_TABLES = {
    "users",
    "roles",
    "permissions",
    "networks",
    "cpts",
    "cpt_versions",
    "cpt_provenance",
    "experts",
    "seed_sets",
    "expert_calibration",
    "contamination_checks",
    "elicitation_sessions",
    "elicitation_session_events",
    "outcomes",
    "calibration_runs",
    "provider_credentials",
}


def test_create_all_builds_every_table() -> None:
    engine = make_engine("sqlite:///:memory:")
    create_all(engine)
    tables = set(inspect(engine).get_table_names())
    assert EXPECTED_TABLES <= tables


def test_cpt_version_and_provenance_roundtrip() -> None:
    engine = make_engine("sqlite:///:memory:")
    create_all(engine)
    Session = make_session_factory(engine)
    with Session() as s:
        net = Network(name="hormuz", topology="latent_regime")
        s.add(net)
        s.flush()

        cpt = Cpt(network_id=net.id, node="Tanker_Incidents")
        s.add(cpt)
        s.flush()

        ver = CptVersion(
            cpt_id=cpt.id,
            version=1,
            values={"('crisis',)": [0.07, 0.31, 0.62]},
            kappa=12.0,
            kappa_level="normal",
        )
        s.add(ver)
        s.flush()

        cpt.current_version_id = ver.id  # resolve the mutual reference
        s.add(
            CptProvenance(
                cpt_version_id=ver.id,
                protocol="cooke",
                kappa_level="normal",
                calibration_score=0.71,
                model_set={"models": ["claude", "gpt", "open-weights"]},
                correlation_note="mean pairwise rho 0.34; effective N 2.1",
                contamination_summary={"perturbation": "passed"},
                is_ai_sourced=True,
            )
        )
        s.commit()

    with Session() as s:
        ver = s.query(CptVersion).one()
        assert ver.kappa_level == "normal"
        assert ver.values["('crisis',)"] == [0.07, 0.31, 0.62]
        prov = s.query(CptProvenance).one()
        assert prov.is_ai_sourced is True
        assert prov.model_set["models"] == ["claude", "gpt", "open-weights"]
        # provenance carries model identity, never a key
        assert "key" not in {k.lower() for k in (prov.model_set or {})}


def test_two_deployment_databases_are_isolated(tmp_path: Path) -> None:
    """Two deployments == two databases; no row in one is visible in the other."""
    url_a = f"sqlite:///{tmp_path / 'client_a.db'}"
    url_b = f"sqlite:///{tmp_path / 'client_b.db'}"
    eng_a, eng_b = make_engine(url_a), make_engine(url_b)
    create_all(eng_a)
    create_all(eng_b)

    SessionA = make_session_factory(eng_a)
    SessionB = make_session_factory(eng_b)
    with SessionA() as s:
        s.add(Network(name="hormuz", topology="latent_regime"))
        s.add(
            ProviderCredentialRow(
                deployment="client_a",
                provider="anthropic",
                source="byok",
                ciphertext="opaque-token",
            )
        )
        s.commit()

    with SessionB() as s:
        assert s.query(Network).count() == 0
        assert s.query(ProviderCredentialRow).count() == 0
