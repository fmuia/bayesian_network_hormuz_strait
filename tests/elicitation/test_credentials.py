"""Tests for credential envelope encryption, redaction, BYOK gating, isolation.

These cover the Layer 0 security-critical validation criteria: a BYOK key is
never present in logs/audit/provenance; BYOK is refused for a non-allowlisted
provider; and one deployment's credentials are never resolvable from another.
"""

from __future__ import annotations

import logging

import pytest

from src.elicitation.config import CredentialSource, DeploymentConfig
from src.elicitation.credentials import (
    BYOKNotAllowedError,
    CredentialResolver,
    InMemoryCredentialStore,
    LocalEnvelopeSecretStore,
    NoCredentialError,
)

SECRET = "sk-super-secret-key-123"


def _store() -> LocalEnvelopeSecretStore:
    return LocalEnvelopeSecretStore(LocalEnvelopeSecretStore.generate_kek())


def _config(name: str, allowlist: list[str] | None = None) -> DeploymentConfig:
    return DeploymentConfig.model_validate(
        {
            "name": name,
            "credentials": {
                "byok": {"enabled": allowlist is not None, "provider_allowlist": allowlist or []}
            },
        }
    )


def test_envelope_roundtrip_and_ciphertext_hides_plaintext() -> None:
    ss = _store()
    token = ss.encrypt(SECRET)
    assert ss.decrypt(token) == SECRET
    assert SECRET not in token  # plaintext must not survive into the token


def test_two_encryptions_differ_but_both_decrypt() -> None:
    """Fresh per-secret data keys mean identical plaintext yields distinct tokens."""
    ss = _store()
    t1, t2 = ss.encrypt(SECRET), ss.encrypt(SECRET)
    assert t1 != t2
    assert ss.decrypt(t1) == ss.decrypt(t2) == SECRET


def test_byok_store_and_resolve_for_allowlisted_provider() -> None:
    ss = _store()
    resolver = CredentialResolver(_config("client_a", ["anthropic"]), InMemoryCredentialStore(), ss)
    resolver.store_byok("anthropic", owner_user_id="u1", api_key=SECRET)
    cred = resolver.resolve("anthropic", owner_user_id="u1")
    assert cred.source is CredentialSource.BYOK
    assert cred.reveal() == SECRET


def test_byok_refused_for_non_allowlisted_provider() -> None:
    resolver = CredentialResolver(_config("client_a", ["anthropic"]), InMemoryCredentialStore(), _store())
    with pytest.raises(BYOKNotAllowedError):
        resolver.store_byok("openai", owner_user_id="u1", api_key=SECRET)


def test_explicit_byok_for_non_allowlisted_provider_raises_on_resolve() -> None:
    resolver = CredentialResolver(_config("client_a", ["anthropic"]), InMemoryCredentialStore(), _store())
    with pytest.raises(BYOKNotAllowedError):
        resolver.resolve("openai", owner_user_id="u1", prefer=CredentialSource.BYOK)


def test_secret_never_appears_in_repr_str_or_audit() -> None:
    resolver = CredentialResolver(_config("client_a", ["anthropic"]), InMemoryCredentialStore(), _store())
    resolver.store_byok("anthropic", owner_user_id="u1", api_key=SECRET)
    cred = resolver.resolve("anthropic", owner_user_id="u1")
    assert SECRET not in repr(cred)
    assert SECRET not in str(cred)
    assert SECRET not in f"{cred}"
    assert SECRET not in str(cred.to_audit_dict())
    # the audit record carries a non-reversible fingerprint, not the key
    assert cred.to_audit_dict()["key_fingerprint"] != SECRET
    assert cred.fingerprint() == cred.fingerprint()  # stable


def test_secret_not_emitted_when_credential_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    resolver = CredentialResolver(_config("client_a", ["anthropic"]), InMemoryCredentialStore(), _store())
    resolver.store_byok("anthropic", owner_user_id="u1", api_key=SECRET)
    cred = resolver.resolve("anthropic", owner_user_id="u1")
    with caplog.at_level(logging.INFO):
        logging.getLogger("test").info("using credential %s", cred)
        logging.getLogger("test").info("audit %s", cred.to_audit_dict())
    assert SECRET not in caplog.text


def test_resolution_falls_back_through_order() -> None:
    """With no BYOK key present, resolution falls through to the deployment key."""
    resolver = CredentialResolver(_config("client_a", ["anthropic"]), InMemoryCredentialStore(), _store())
    resolver.store_deployment_key("anthropic", "sk-deployment")
    cred = resolver.resolve("anthropic", owner_user_id="u1")
    assert cred.source is CredentialSource.DEPLOYMENT_KEY
    assert cred.reveal() == "sk-deployment"


def test_no_credential_raises() -> None:
    resolver = CredentialResolver(_config("client_a", ["anthropic"]), InMemoryCredentialStore(), _store())
    with pytest.raises(NoCredentialError):
        resolver.resolve("anthropic", owner_user_id="u1")


def test_deployment_isolation_even_with_shared_store() -> None:
    """A credential stored under one deployment is invisible to another."""
    ss = _store()
    shared = InMemoryCredentialStore()
    a = CredentialResolver(_config("client_a", ["anthropic"]), shared, ss)
    b = CredentialResolver(_config("client_b", ["anthropic"]), shared, ss)
    a.store_byok("anthropic", owner_user_id="u1", api_key=SECRET)
    assert a.resolve("anthropic", owner_user_id="u1").reveal() == SECRET
    with pytest.raises(NoCredentialError):
        b.resolve("anthropic", owner_user_id="u1")


def test_byok_scoped_per_user() -> None:
    resolver = CredentialResolver(_config("client_a", ["anthropic"]), InMemoryCredentialStore(), _store())
    resolver.store_byok("anthropic", owner_user_id="u1", api_key=SECRET)
    with pytest.raises(NoCredentialError):
        resolver.resolve("anthropic", owner_user_id="u2", prefer=CredentialSource.BYOK)
