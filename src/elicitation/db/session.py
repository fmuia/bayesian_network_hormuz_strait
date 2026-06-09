"""Engine and session helpers.

One engine per deployment, pointed at that deployment's isolated database
(``DeploymentConfig.database.url``). ``create_all`` is convenient for tests and
local development; production schema changes go through Alembic migrations.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from .schema import Base


def make_engine(url: str) -> Engine:
    """Create an Engine for a deployment database URL.

    A bare in-memory SQLite URL (``sqlite://`` / ``sqlite:///:memory:``) uses a
    StaticPool so every session shares one in-memory database — useful for the
    demo app and tests.
    """
    if url.startswith("sqlite"):
        if url in ("sqlite://", "sqlite:///:memory:"):
            return create_engine(
                "sqlite://",
                future=True,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        return create_engine(url, future=True, connect_args={"check_same_thread": False})
    return create_engine(url, future=True)


def create_all(engine: Engine) -> None:
    """Create every table. For tests/dev; production uses Alembic."""
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a configured session factory bound to ``engine``."""
    return sessionmaker(bind=engine, expire_on_commit=False, class_=Session)
