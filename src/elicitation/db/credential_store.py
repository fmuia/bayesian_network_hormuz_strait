"""Database-backed credential store.

Implements the :class:`~src.elicitation.credentials.resolver.CredentialStore`
protocol over the ``provider_credentials`` table, honouring the same
deployment-scoping contract the in-memory store does: reads are filtered to the
requested deployment, so credentials never leak across deployments. ``put``
upserts on the unique scope ``(deployment, provider, source, owner_user_id)``,
so re-storing a key rotates it in place. Only ciphertext and a non-reversible
fingerprint are persisted — never the plaintext key.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config.models import CredentialSource
from ..credentials.resolver import StoredCredential
from .schema import ProviderCredentialRow


class SqlCredentialStore:
    """A :class:`CredentialStore` backed by a SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def _query(self, deployment: str, provider: str, source: CredentialSource, owner_user_id):
        stmt = select(ProviderCredentialRow).where(
            ProviderCredentialRow.deployment == deployment,
            ProviderCredentialRow.provider == provider.strip().lower(),
            ProviderCredentialRow.source == source.value,
        )
        if source is CredentialSource.BYOK:
            stmt = stmt.where(ProviderCredentialRow.owner_user_id == owner_user_id)
        else:
            stmt = stmt.where(ProviderCredentialRow.owner_user_id.is_(None))
        return stmt

    def get(
        self,
        deployment: str,
        provider: str,
        source: CredentialSource,
        owner_user_id: str | None = None,
    ) -> StoredCredential | None:
        stmt = self._query(deployment, provider, source, owner_user_id).where(
            ProviderCredentialRow.is_active.is_(True)
        )
        row = self._session.scalars(stmt).first()
        if row is None:
            return None
        return StoredCredential(
            deployment=row.deployment,
            provider=row.provider,
            source=CredentialSource(row.source),
            ciphertext=row.ciphertext,
            owner_user_id=row.owner_user_id,
            expires_at=row.expires_at,
            key_fingerprint=row.key_fingerprint,
        )

    def put(self, stored: StoredCredential) -> None:
        provider = stored.provider.strip().lower()
        existing = self._session.scalars(
            self._query(stored.deployment, provider, stored.source, stored.owner_user_id)
        ).first()
        if existing is not None:
            existing.ciphertext = stored.ciphertext
            existing.key_fingerprint = stored.key_fingerprint
            existing.expires_at = stored.expires_at
            existing.is_active = True
            existing.rotated_at = datetime.now(timezone.utc)
        else:
            self._session.add(
                ProviderCredentialRow(
                    deployment=stored.deployment,
                    provider=provider,
                    source=stored.source.value,
                    owner_user_id=stored.owner_user_id,
                    ciphertext=stored.ciphertext,
                    key_fingerprint=stored.key_fingerprint,
                    expires_at=stored.expires_at,
                )
            )
        self._session.commit()


__all__ = ["SqlCredentialStore"]
