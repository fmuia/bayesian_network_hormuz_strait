"""Resolve LLM-provider credentials from three sources, behind one interface.

A :class:`CredentialResolver` is bound to a single deployment and serves a
:class:`~src.elicitation.credentials.models.ProviderCredential` from whichever
source the deployment policy permits — bring-your-own-key, a deployment key, or
an OAuth token. Two invariants matter:

* **Deployment scoping.** The resolver only ever reads rows for its own
  deployment, so one deployment's credentials are never resolvable from another.
* **BYOK gating.** A BYOK key is accepted (at write time) and used (at read
  time) only for providers on the deployment's allowlist; otherwise it is
  refused. This is enforced here, not left to the caller.

The encrypted rows live behind the :class:`CredentialStore` protocol. An
in-memory implementation is provided for tests; the database-backed
implementation arrives with the schema and must honour the same
deployment-scoping contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ..config.models import CredentialSource, DeploymentConfig
from .models import ProviderCredential
from .secret_store import SecretStore


class CredentialError(Exception):
    """Base class for credential-resolution errors."""


class BYOKNotAllowedError(CredentialError):
    """A BYOK key was offered or requested for a non-allowlisted provider."""


class NoCredentialError(CredentialError):
    """No credential could be resolved for the provider in this deployment."""


@dataclass(frozen=True)
class StoredCredential:
    """An encrypted credential as persisted, scoped by deployment.

    ``ciphertext`` is a :class:`SecretStore` token — never plaintext.
    ``key_fingerprint`` is a non-reversible digest of the key, for audit
    correlation only.
    """

    deployment: str
    provider: str
    source: CredentialSource
    ciphertext: str
    owner_user_id: str | None = None
    expires_at: datetime | None = None
    key_fingerprint: str | None = None


def _fingerprint(secret: str) -> str:
    import hashlib

    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:12]


class CredentialStore(Protocol):
    """Persistence for encrypted credentials. Implementations MUST scope reads
    to the requested ``deployment`` and never leak across deployments."""

    def get(
        self,
        deployment: str,
        provider: str,
        source: CredentialSource,
        owner_user_id: str | None = None,
    ) -> StoredCredential | None: ...

    def put(self, stored: StoredCredential) -> None: ...


class InMemoryCredentialStore:
    """A dict-backed :class:`CredentialStore` for tests and local development."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str, CredentialSource, str | None], StoredCredential] = {}

    @staticmethod
    def _key(
        deployment: str, provider: str, source: CredentialSource, owner: str | None
    ) -> tuple[str, str, CredentialSource, str | None]:
        return (deployment, provider.strip().lower(), source, owner)

    def get(
        self,
        deployment: str,
        provider: str,
        source: CredentialSource,
        owner_user_id: str | None = None,
    ) -> StoredCredential | None:
        return self._rows.get(self._key(deployment, provider, source, owner_user_id))

    def put(self, stored: StoredCredential) -> None:
        self._rows[
            self._key(stored.deployment, stored.provider, stored.source, stored.owner_user_id)
        ] = stored


class CredentialResolver:
    """Resolve provider credentials for one deployment."""

    def __init__(
        self,
        config: DeploymentConfig,
        store: CredentialStore,
        secret_store: SecretStore,
    ) -> None:
        self._config = config
        self._store = store
        self._secret = secret_store

    @property
    def deployment(self) -> str:
        return self._config.name

    # -- writing credentials -------------------------------------------------

    def store_byok(self, provider: str, owner_user_id: str, api_key: str) -> None:
        """Encrypt and persist a user's BYOK key. Refused if not allowlisted."""
        if not self._config.credentials.byok.allows(provider):
            raise BYOKNotAllowedError(
                f"BYOK is not permitted for provider {provider!r} in deployment "
                f"{self.deployment!r} (provider not on the allowlist, or BYOK disabled)"
            )
        self._store.put(
            StoredCredential(
                deployment=self.deployment,
                provider=provider,
                source=CredentialSource.BYOK,
                ciphertext=self._secret.encrypt(api_key),
                owner_user_id=owner_user_id,
                key_fingerprint=_fingerprint(api_key),
            )
        )

    def store_deployment_key(self, provider: str, api_key: str) -> None:
        """Encrypt and persist the operator-set deployment key for a provider."""
        self._store.put(
            StoredCredential(
                deployment=self.deployment,
                provider=provider,
                source=CredentialSource.DEPLOYMENT_KEY,
                ciphertext=self._secret.encrypt(api_key),
                key_fingerprint=_fingerprint(api_key),
            )
        )

    def store_oauth_token(
        self,
        provider: str,
        token: str,
        owner_user_id: str | None = None,
        expires_at: datetime | None = None,
    ) -> None:
        """Encrypt and persist an OAuth-brokered token for a provider."""
        self._store.put(
            StoredCredential(
                deployment=self.deployment,
                provider=provider,
                source=CredentialSource.OAUTH,
                ciphertext=self._secret.encrypt(token),
                owner_user_id=owner_user_id,
                expires_at=expires_at,
                key_fingerprint=_fingerprint(token),
            )
        )

    # -- resolving credentials ----------------------------------------------

    def resolve(
        self,
        provider: str,
        owner_user_id: str | None = None,
        prefer: CredentialSource | None = None,
    ) -> ProviderCredential:
        """Return the first available credential for ``provider``.

        Sources are tried in ``prefer`` order if given, else the deployment's
        configured ``resolution_order``. Requesting BYOK explicitly for a
        non-allowlisted provider raises; during normal fallback that source is
        simply skipped.
        """
        order = [prefer] if prefer is not None else self._config.credentials.resolution_order
        for source in order:
            if source is CredentialSource.BYOK:
                if not self._config.credentials.byok.allows(provider):
                    if prefer is CredentialSource.BYOK:
                        raise BYOKNotAllowedError(
                            f"BYOK is not permitted for provider {provider!r} in "
                            f"deployment {self.deployment!r}"
                        )
                    continue
                stored = self._store.get(
                    self.deployment, provider, CredentialSource.BYOK, owner_user_id
                )
            else:
                stored = self._store.get(self.deployment, provider, source, None)

            if stored is None:
                continue

            return ProviderCredential(
                provider=provider,
                source=source,
                deployment=self.deployment,
                secret=self._secret.decrypt(stored.ciphertext),
                owner_user_id=stored.owner_user_id,
                expires_at=stored.expires_at,
            )

        raise NoCredentialError(
            f"no credential available for provider {provider!r} in deployment "
            f"{self.deployment!r}"
        )
