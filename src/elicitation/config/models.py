"""Per-deployment configuration models.

Each deployment of the elicitation platform runs an isolated stack with its
own configuration: which network and topology are active, which CPTs are in
scope for elicitation, the database location, and the LLM-provider credential
policy — including the bring-your-own-key (BYOK) provider allowlist.

These are pure pydantic models with no I/O; ``loader.py`` reads YAML into them.
The BYOK allowlist is the security-relevant field: a provider absent from it
(or an empty list) means BYOK is refused for that provider — see
``src.elicitation.credentials``.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CredentialSource(str, Enum):
    """Where an LLM provider credential comes from.

    ``DEPLOYMENT_KEY`` — a key set by the deployment operator.
    ``OAUTH`` — an OAuth-brokered token.
    ``BYOK`` — a key supplied by an individual user (bring your own key).
    """

    DEPLOYMENT_KEY = "deployment_key"
    OAUTH = "oauth"
    BYOK = "byok"


class BYOKPolicy(BaseModel):
    """Per-deployment bring-your-own-key policy.

    BYOK routes the deployment's source material to the chosen provider under
    the user's account and terms, so it is a data-residency decision, not only
    a security one. It is therefore gated to an explicit provider allowlist and
    can be disabled wholesale for high-sensitivity deployments.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    provider_allowlist: list[str] = Field(default_factory=list)

    @field_validator("provider_allowlist")
    @classmethod
    def _normalise(cls, value: list[str]) -> list[str]:
        return [p.strip().lower() for p in value if p.strip()]

    def allows(self, provider: str) -> bool:
        """True iff BYOK is enabled and ``provider`` is on the allowlist."""
        return self.enabled and provider.strip().lower() in self.provider_allowlist


class SecretStoreConfig(BaseModel):
    """How credential secrets are encrypted at rest.

    ``backend = "local_envelope"`` is the development adapter: a local
    key-encryption key (KEK) read from ``kek_env`` wraps a per-secret data key
    (envelope encryption). Production swaps in a managed KMS backend without
    changing the stored-token format. The KEK itself never lives in config or
    the database — only the name of the environment variable that holds it.
    """

    model_config = ConfigDict(extra="forbid")

    backend: str = "local_envelope"
    kek_env: str = "ELICITATION_KEK"


class CredentialsConfig(BaseModel):
    """LLM provider credential policy for the deployment."""

    model_config = ConfigDict(extra="forbid")

    byok: BYOKPolicy = Field(default_factory=BYOKPolicy)
    secret_store: SecretStoreConfig = Field(default_factory=SecretStoreConfig)
    # Order in which sources are tried when resolving a provider credential.
    resolution_order: list[CredentialSource] = Field(
        default_factory=lambda: [
            CredentialSource.BYOK,
            CredentialSource.DEPLOYMENT_KEY,
            CredentialSource.OAUTH,
        ]
    )


class DatabaseConfig(BaseModel):
    """Per-deployment database location. One database == one deployment."""

    model_config = ConfigDict(extra="forbid")

    url: str = "sqlite:///elicitation.db"


class DeploymentConfig(BaseModel):
    """The full per-deployment configuration.

    ``name`` is the deployment identifier and is used to scope credentials and
    data — no row in one deployment is ever visible to another (the platform
    has no ``tenant_id``; isolation is at the database/deployment boundary).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    # TODO(pack-separation): default hardcodes the Hormuz pack id — default to
    # packs.registry.DEFAULT_PACK instead of the literal "hormuz".
    network: str = "hormuz"
    topology: str = "latent_regime"
    in_scope_cpts: list[str] = Field(default_factory=list)
    branding: dict[str, str] = Field(default_factory=dict)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    credentials: CredentialsConfig = Field(default_factory=CredentialsConfig)

    @field_validator("name")
    @classmethod
    def _non_empty_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("deployment name must be non-empty")
        return value.strip()
