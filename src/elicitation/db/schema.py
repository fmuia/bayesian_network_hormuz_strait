"""Database schema for the elicitation platform (Plan 4, Layer 0).

One database is one deployment: there is no ``tenant_id`` column anywhere, and
isolation is at the database boundary. The tables below are the
elicitation-specific schema. The translator audit-log tables (``articles``,
``translations``, ``analyst_actions``, ``sources``) are owned by Plan 2 and are
*not* redefined here; cross-plan references are intentionally soft (plain id
columns, no hard ForeignKey) so the two schemas can be migrated independently.

Conventions:
* SQLAlchemy 2.0 declarative (`Mapped` / `mapped_column`).
* A constraint naming convention so Alembic autogenerates stable names.
* Flexible/structured fields (CPT value vectors, model sets, config blobs,
  references) use the portable ``JSON`` type.
* Credential secrets are stored only as envelope-encrypted ciphertext in
  ``provider_credentials.ciphertext`` — never plaintext (see
  ``src.elicitation.credentials``).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


# --------------------------------------------------------------------------- #
# RBAC primitives
# --------------------------------------------------------------------------- #


class Role(Base):
    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


class Permission(Base):
    __tablename__ = "permissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(128), unique=True)
    email: Mapped[str | None] = mapped_column(String(256), nullable=True)
    role_id: Mapped[int | None] = mapped_column(ForeignKey("roles.id"), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# --------------------------------------------------------------------------- #
# Networks and CPTs
# --------------------------------------------------------------------------- #


class Network(Base):
    __tablename__ = "networks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    topology: Mapped[str] = mapped_column(String(64))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    cpts: Mapped[list["Cpt"]] = relationship(back_populates="network")


class Cpt(Base):
    """A CPT's current pointer, identified by network + node.

    The actual probability values live in ``cpt_versions``; ``current_version_id``
    points at the version in force.
    """

    __tablename__ = "cpts"
    __table_args__ = (UniqueConstraint("network_id", "node", name="network_node"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    network_id: Mapped[int] = mapped_column(ForeignKey("networks.id"))
    node: Mapped[str] = mapped_column(String(128))
    # cpts <-> cpt_versions are mutually dependent (a CPT points at its current
    # version; a version belongs to a CPT). use_alter emits this FK as a
    # post-create ALTER so DDL ordering is valid on Postgres as well as SQLite.
    current_version_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "cpt_versions.id",
            use_alter=True,
            name="fk_cpts_current_version_id_cpt_versions",
        ),
        nullable=True,
    )

    network: Mapped["Network"] = relationship(back_populates="cpts")
    versions: Mapped[list["CptVersion"]] = relationship(
        back_populates="cpt", foreign_keys="CptVersion.cpt_id"
    )


class CptVersion(Base):
    """A historical CPT value with full audit trail.

    ``values`` is the JSON-encoded CPT (parent-config -> probability vector).
    ``kappa`` and ``kappa_level`` carry the per-CPT calibration concentration
    (Plan 4, finding M3) that the inference layer consumes.
    """

    __tablename__ = "cpt_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cpt_id: Mapped[int] = mapped_column(ForeignKey("cpts.id"))
    version: Mapped[int] = mapped_column(Integer)
    values: Mapped[dict] = mapped_column(JSON)
    kappa: Mapped[float | None] = mapped_column(Float, nullable=True)
    kappa_level: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    cpt: Mapped["Cpt"] = relationship(back_populates="versions", foreign_keys=[cpt_id])
    provenance: Mapped["CptProvenance | None"] = relationship(back_populates="cpt_version")


class CptProvenance(Base):
    """Per-CPT-version provenance.

    For AI-sourced CPTs this is a hard defensibility record: protocol, κ level,
    calibration score, the set of base models in the panel, an inter-agent
    correlation note, and a contamination-probe summary. The provider key is
    never stored here — only model identity.
    """

    __tablename__ = "cpt_provenance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cpt_version_id: Mapped[int] = mapped_column(ForeignKey("cpt_versions.id"), unique=True)
    protocol: Mapped[str] = mapped_column(String(32))
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("elicitation_sessions.id"), nullable=True
    )
    elicited_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    references: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    kappa_level: Mapped[str | None] = mapped_column(String(16), nullable=True)
    calibration_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_set: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    correlation_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    contamination_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_ai_sourced: Mapped[bool] = mapped_column(Boolean, default=False)
    human_signed_off_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )

    cpt_version: Mapped["CptVersion"] = relationship(back_populates="provenance")


# --------------------------------------------------------------------------- #
# Experts and calibration
# --------------------------------------------------------------------------- #


class Expert(Base):
    """A registered expert — human or AI.

    An AI expert's identity is the tuple ``(base_model, role, config)``;
    calibration is measured per tuple (Plan 4, decision B.16). For a human,
    ``base_model`` is null and ``role`` is optional.
    """

    __tablename__ = "experts"
    __table_args__ = (
        UniqueConstraint("kind", "base_model", "role", "config_fingerprint", name="identity"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(8))  # "human" | "ai"
    display_name: Mapped[str] = mapped_column(String(256))
    base_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    role: Mapped[str | None] = mapped_column(String(64), nullable=True)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    config_fingerprint: Mapped[str | None] = mapped_column(String(16), nullable=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    calibration: Mapped[list["ExpertCalibration"]] = relationship(back_populates="expert")


class SeedSet(Base):
    """A per-deployment calibration question set.

    Seeds must probe the same judgment as the targets (relevance constraint).
    ``questions`` holds the seed items with their resolution dates and source
    provenance; ``resolution_dates`` and contamination handling let AI-expert
    calibration distinguish recall from reasoning.
    """

    __tablename__ = "seed_sets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    domain: Mapped[str | None] = mapped_column(String(128), nullable=True)
    questions: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ExpertCalibration(Base):
    """Per-(expert, seed-set) performance: the Cooke calibration & information
    scores, the resulting weight, and the κ level the expert may contribute."""

    __tablename__ = "expert_calibration"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    expert_id: Mapped[int] = mapped_column(ForeignKey("experts.id"))
    seed_set_id: Mapped[int] = mapped_column(ForeignKey("seed_sets.id"))
    calibration_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    information_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    kappa_cap: Mapped[str | None] = mapped_column(String(16), nullable=True)
    scoring_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    expert: Mapped["Expert"] = relationship(back_populates="calibration")


class ContaminationCheck(Base):
    """Per-AI-expert, per-seed contamination-probe result (Plan 4, Layer 3)."""

    __tablename__ = "contamination_checks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    expert_id: Mapped[int] = mapped_column(ForeignKey("experts.id"))
    seed_set_id: Mapped[int] = mapped_column(ForeignKey("seed_sets.id"))
    seed_ref: Mapped[str] = mapped_column(String(128))
    probe: Mapped[str] = mapped_column(String(32))  # source_attribution | perturbation | split | variance
    flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# --------------------------------------------------------------------------- #
# Elicitation sessions (resumable state machine)
# --------------------------------------------------------------------------- #


class ElicitationSession(Base):
    __tablename__ = "elicitation_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    network_id: Mapped[int] = mapped_column(ForeignKey("networks.id"))
    node: Mapped[str] = mapped_column(String(128))
    protocol: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default="in_progress")
    inputs: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    aggregated_output: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    events: Mapped[list["ElicitationSessionEvent"]] = relationship(back_populates="session")


class ElicitationSessionEvent(Base):
    """State-machine events that make a protocol run resumable across restarts."""

    __tablename__ = "elicitation_session_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("elicitation_sessions.id"))
    seq: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    session: Mapped["ElicitationSession"] = relationship(back_populates="events")


# --------------------------------------------------------------------------- #
# Calibration tracking over time
# --------------------------------------------------------------------------- #


class Outcome(Base):
    """A realised intermediate-node state, for Tier 2/3 calibration.

    ``article_id`` is a soft reference into Plan 2's audit log (no hard FK).
    """

    __tablename__ = "outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    network_id: Mapped[int] = mapped_column(ForeignKey("networks.id"))
    node: Mapped[str] = mapped_column(String(128))
    realised_state: Mapped[str] = mapped_column(String(128))
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    article_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


class CalibrationRun(Base):
    """A scheduled evaluation result (Brier scores, reliability, Bayes factors)."""

    __tablename__ = "calibration_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    network_id: Mapped[int] = mapped_column(ForeignKey("networks.id"))
    tier: Mapped[int] = mapped_column(Integer)
    results: Mapped[dict] = mapped_column(JSON)
    ran_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# --------------------------------------------------------------------------- #
# LLM provider credentials
# --------------------------------------------------------------------------- #


class ProviderCredentialRow(Base):
    """An envelope-encrypted LLM-provider credential, scoped to this deployment.

    ``ciphertext`` is a secret-store token; the plaintext key is never stored.
    ``key_fingerprint`` is a non-reversible digest for audit correlation only.
    ``deployment`` scopes the row — the resolver only reads rows for its own
    deployment, so credentials never leak across deployments.
    """

    __tablename__ = "provider_credentials"
    __table_args__ = (
        UniqueConstraint(
            "deployment", "provider", "source", "owner_user_id", name="scope"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    deployment: Mapped[str] = mapped_column(String(128), index=True)
    provider: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(16))  # deployment_key | oauth | byok
    # An opaque per-deployment user identifier (username / SSO subject / stringified
    # users.id), matching the credential layer's str owner id. Soft reference, no FK.
    owner_user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ciphertext: Mapped[str] = mapped_column(Text)
    key_fingerprint: Mapped[str | None] = mapped_column(String(16), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    rotated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


__all__ = [
    "Base",
    "Role",
    "Permission",
    "User",
    "Network",
    "Cpt",
    "CptVersion",
    "CptProvenance",
    "Expert",
    "SeedSet",
    "ExpertCalibration",
    "ContaminationCheck",
    "ElicitationSession",
    "ElicitationSessionEvent",
    "Outcome",
    "CalibrationRun",
    "ProviderCredentialRow",
]
