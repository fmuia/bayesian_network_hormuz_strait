"""The resolved-credential value object.

A :class:`ProviderCredential` is what the agents receive when they need to call
an LLM provider. The secret is wrapped in a pydantic ``SecretStr`` so it never
appears in ``repr``/``str``, in logs, or in ``model_dump()`` — the plaintext is
reachable only through the explicit :meth:`reveal`, which callers use solely at
the HTTP boundary to the provider. Provenance and audit records use
:meth:`to_audit_dict`, which records the *model identity* and a non-reversible
key fingerprint but never the key itself.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from pydantic import BaseModel, ConfigDict, SecretStr

from ..config.models import CredentialSource


class ProviderCredential(BaseModel):
    """A resolved LLM-provider credential, scoped to one deployment."""

    model_config = ConfigDict(frozen=True)

    provider: str
    source: CredentialSource
    deployment: str
    secret: SecretStr
    owner_user_id: str | None = None
    expires_at: datetime | None = None

    def reveal(self) -> str:
        """Return the plaintext secret. Use only at the provider boundary."""
        return self.secret.get_secret_value()

    def fingerprint(self) -> str:
        """A short, non-reversible fingerprint of the secret.

        Lets provenance correlate *which* key produced a CPT without ever
        recording the key — the "model identity recorded, key never" rule.
        """
        return hashlib.sha256(self.reveal().encode("utf-8")).hexdigest()[:12]

    def to_audit_dict(self) -> dict[str, str | None]:
        """A log- and provenance-safe description. Never includes the secret."""
        return {
            "provider": self.provider,
            "source": self.source.value,
            "deployment": self.deployment,
            "owner_user_id": self.owner_user_id,
            "key_fingerprint": self.fingerprint(),
        }

    def __repr__(self) -> str:
        return (
            f"ProviderCredential(provider={self.provider!r}, "
            f"source={self.source.value!r}, deployment={self.deployment!r}, "
            f"owner_user_id={self.owner_user_id!r}, secret=SecretStr('**********'))"
        )

    __str__ = __repr__
