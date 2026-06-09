"""LLM-provider credentials: resolution, envelope encryption, redaction.

See ``docs/04_elicitation_tool_plan.md`` Layer 0 and decision B.15. The secret
is never logged or written to provenance; only the model identity and a
non-reversible key fingerprint are recorded.
"""

from __future__ import annotations

from .models import ProviderCredential
from .resolver import (
    BYOKNotAllowedError,
    CredentialError,
    CredentialResolver,
    CredentialStore,
    InMemoryCredentialStore,
    NoCredentialError,
    StoredCredential,
)
from .secret_store import LocalEnvelopeSecretStore, SecretStore

__all__ = [
    "ProviderCredential",
    "CredentialResolver",
    "CredentialStore",
    "InMemoryCredentialStore",
    "StoredCredential",
    "CredentialError",
    "BYOKNotAllowedError",
    "NoCredentialError",
    "SecretStore",
    "LocalEnvelopeSecretStore",
]
