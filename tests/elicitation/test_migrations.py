"""Alembic migration round-trip: upgrade creates every table, downgrade removes them."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from tests.elicitation.test_schema import EXPECTED_TABLES

_MIGRATIONS = "src/elicitation/db/migrations"


def _alembic_config(db_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", _MIGRATIONS)
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


def test_migration_upgrade_then_downgrade_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "migrate.db"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("ELICITATION_DB_URL", url)
    cfg = _alembic_config(url)

    command.upgrade(cfg, "head")
    tables = set(inspect(create_engine(url)).get_table_names())
    assert EXPECTED_TABLES <= tables
    assert "alembic_version" in tables

    command.downgrade(cfg, "base")
    after = set(inspect(create_engine(url)).get_table_names())
    # only Alembic's bookkeeping table remains; every elicitation table is gone
    assert after <= {"alembic_version"}
