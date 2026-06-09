"""Tests for the database-backed credential store (closes the Layer 0 gap).

The same security contract the in-memory store satisfies must hold against the
real ``provider_credentials`` table: redaction, BYOK allowlist gating,
per-deployment isolation, and rotation in place.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from src.elicitation.config import CredentialSource, DeploymentConfig
from src.elicitation.credentials import (
    BYOKNotAllowedError,
    CredentialResolver,
    LocalEnvelopeSecretStore,
    NoCredentialError,
)
from src.elicitation.db import SqlCredentialStore, create_all, make_engine, make_session_factory
from src.elicitation.db.schema import ProviderCredentialRow

SECRET = "sk-sql-secret-456"


def _config(name: str, allowlist: list[str] | None = None) -> DeploymentConfig:
    return DeploymentConfig.model_validate(
        {"name": name, "credentials": {"byok": {"enabled": allowlist is not None, "provider_allowlist": allowlist or []}}}
    )


@pytest.fixture()
def session():
    engine = make_engine("sqlite://")  # shared in-memory
    create_all(engine)
    Session = make_session_factory(engine)
    with Session() as s:
        yield s


def _resolver(cfg, session) -> CredentialResolver:
    secret = LocalEnvelopeSecretStore(LocalEnvelopeSecretStore.generate_kek())
    return CredentialResolver(cfg, SqlCredentialStore(session), secret)


def test_store_and_resolve_byok_through_database(session) -> None:
    resolver = _resolver(_config("client_a", ["anthropic"]), session)
    resolver.store_byok("anthropic", owner_user_id="u1", api_key=SECRET)
    cred = resolver.resolve("anthropic", owner_user_id="u1")
    assert cred.source is CredentialSource.BYOK
    assert cred.reveal() == SECRET


def test_database_row_holds_ciphertext_and_fingerprint_not_plaintext(session) -> None:
    resolver = _resolver(_config("client_a", ["anthropic"]), session)
    resolver.store_byok("anthropic", owner_user_id="u1", api_key=SECRET)
    row = session.scalars(select(ProviderCredentialRow)).one()
    assert SECRET not in row.ciphertext
    assert row.key_fingerprint and row.key_fingerprint != SECRET
    assert len(row.key_fingerprint) == 12


def test_non_allowlisted_byok_refused(session) -> None:
    resolver = _resolver(_config("client_a", ["anthropic"]), session)
    with pytest.raises(BYOKNotAllowedError):
        resolver.store_byok("openai", owner_user_id="u1", api_key=SECRET)
    assert session.scalars(select(ProviderCredentialRow)).first() is None


def test_deployment_isolation_in_one_database(session) -> None:
    a = _resolver(_config("client_a", ["anthropic"]), session)
    b = _resolver(_config("client_b", ["anthropic"]), session)
    a.store_byok("anthropic", owner_user_id="u1", api_key=SECRET)
    assert a.resolve("anthropic", owner_user_id="u1").reveal() == SECRET
    with pytest.raises(NoCredentialError):
        b.resolve("anthropic", owner_user_id="u1")


def test_put_rotates_in_place_on_unique_scope(session) -> None:
    resolver = _resolver(_config("client_a", ["anthropic"]), session)
    resolver.store_byok("anthropic", owner_user_id="u1", api_key="sk-old")
    resolver.store_byok("anthropic", owner_user_id="u1", api_key="sk-new")
    rows = session.scalars(select(ProviderCredentialRow)).all()
    assert len(rows) == 1  # rotated in place, not duplicated
    assert resolver.resolve("anthropic", owner_user_id="u1").reveal() == "sk-new"
    assert rows[0].rotated_at is not None


def test_resolution_falls_back_to_deployment_key(session) -> None:
    resolver = _resolver(_config("client_a", ["anthropic"]), session)
    resolver.store_deployment_key("anthropic", "sk-deployment")
    cred = resolver.resolve("anthropic", owner_user_id="u1")
    assert cred.source is CredentialSource.DEPLOYMENT_KEY
    assert cred.reveal() == "sk-deployment"


def test_byok_scoped_per_user_in_database(session) -> None:
    resolver = _resolver(_config("client_a", ["anthropic"]), session)
    resolver.store_byok("anthropic", owner_user_id="u1", api_key=SECRET)
    with pytest.raises(NoCredentialError):
        resolver.resolve("anthropic", owner_user_id="u2", prefer=CredentialSource.BYOK)
