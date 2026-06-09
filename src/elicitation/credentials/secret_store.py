"""Secret storage with envelope encryption.

Envelope encryption: a fresh random *data encryption key* (DEK) encrypts each
secret, and the DEK is itself wrapped by a *key encryption key* (KEK). Only the
wrapped DEK and the ciphertext are persisted — never the plaintext, never an
unwrapped DEK. The development backend keeps the KEK in an environment variable;
a production backend replaces the wrap/unwrap step with a managed KMS call
without changing the stored-token format.
"""

from __future__ import annotations

import base64
import json
import os
from abc import ABC, abstractmethod

from cryptography.fernet import Fernet


class SecretStore(ABC):
    """Encrypt/decrypt secrets at rest. Tokens are opaque, safe to persist."""

    @abstractmethod
    def encrypt(self, plaintext: str) -> str:
        """Return an opaque token that :meth:`decrypt` reverses."""

    @abstractmethod
    def decrypt(self, token: str) -> str:
        """Recover the plaintext from a token produced by :meth:`encrypt`."""


class LocalEnvelopeSecretStore(SecretStore):
    """Envelope encryption backed by a local KEK (development / self-hosted).

    The KEK is a Fernet key (urlsafe-base64, 32 bytes). In production the KEK
    lives in a KMS and the wrap/unwrap below become KMS ``Encrypt``/``Decrypt``
    calls; the token shape is unchanged, so stored credentials migrate cleanly.
    """

    def __init__(self, kek: bytes | str) -> None:
        self._kek = Fernet(kek.encode("ascii") if isinstance(kek, str) else kek)

    @classmethod
    def from_env(cls, var: str) -> "LocalEnvelopeSecretStore":
        """Build from a KEK held in the ``var`` environment variable."""
        key = os.environ.get(var)
        if not key:
            raise KeyError(f"key-encryption-key environment variable {var!r} is not set")
        return cls(key)

    @staticmethod
    def generate_kek() -> bytes:
        """Generate a fresh KEK. Store it in a secret manager, not in code."""
        return Fernet.generate_key()

    def encrypt(self, plaintext: str) -> str:
        dek = Fernet.generate_key()
        ciphertext = Fernet(dek).encrypt(plaintext.encode("utf-8"))
        wrapped_dek = self._kek.encrypt(dek)
        envelope = {
            "v": 1,
            "edek": base64.b64encode(wrapped_dek).decode("ascii"),
            "ct": base64.b64encode(ciphertext).decode("ascii"),
        }
        return base64.b64encode(json.dumps(envelope).encode("utf-8")).decode("ascii")

    def decrypt(self, token: str) -> str:
        envelope = json.loads(base64.b64decode(token))
        wrapped_dek = base64.b64decode(envelope["edek"])
        ciphertext = base64.b64decode(envelope["ct"])
        dek = self._kek.decrypt(wrapped_dek)
        return Fernet(dek).decrypt(ciphertext).decode("utf-8")
