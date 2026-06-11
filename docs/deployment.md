# Deploying the elicitation platform

This guide stands up a new **deployment** of the elicitation platform (Plan 4,
Layer 0). Each deployment is **single-tenant**: its own database, its own users,
its own configuration, with no data co-mingling across deployments. There is no
`tenant_id` anywhere — isolation is at the database boundary.

> Status: Layer 0 (data model, configuration, credentials, migrations). Later
> layers add the Cooke protocol, AI panels, the UI, and inference integration —
> see [`04_elicitation_tool_plan.md`](04_elicitation_tool_plan.md).

## 1. Prerequisites

The dependencies are declared in `pixi.toml` / `requirements.txt`:

```bash
pixi install          # or: pip install -r requirements.txt
```

Layer 0 adds `sqlalchemy`, `alembic`, `pyyaml`, and `cryptography`.

## 2. Write the deployment configuration

Create a YAML file per deployment (e.g. `deployments/client_a.yaml`). Secrets
may be referenced as `${ENV_VAR}` and are expanded from the environment at load
time, so they never live in the file.

```yaml
name: client_a                     # deployment id; scopes all data + credentials
network: hormuz
topology: latent_regime
in_scope_cpts: []                  # empty = all CPTs in scope for elicitation
database:
  url: ${CLIENT_A_DB_URL}          # e.g. postgresql+psycopg://user:pass@host/client_a
credentials:
  secret_store:
    backend: local_envelope        # KMS-backed in production
    kek_env: CLIENT_A_KEK          # env var holding the key-encryption key
  byok:
    enabled: true                  # set false to disable BYOK entirely
    provider_allowlist:            # BYOK permitted ONLY for these providers
      - anthropic
      - openai
  resolution_order: [byok, deployment_key, oauth]
```

Load and validate it in code with:

```python
from src.elicitation.config import load_deployment_config
cfg = load_deployment_config("deployments/client_a.yaml")
```

### BYOK is a data-residency decision

A bring-your-own-key credential routes this deployment's source material to the
chosen provider under the user's own account and terms. Treat the
`provider_allowlist` as a governance control, not a convenience: leave it empty
(or `enabled: false`) for high-sensitivity deployments. A key for a provider not
on the allowlist is **refused at write time and at resolve time**.

## 3. Provision the key-encryption key (KEK)

Credential secrets are stored with **envelope encryption**: a fresh data key
encrypts each secret, and the KEK wraps the data key. Only the wrapped data key
and ciphertext are persisted — never a plaintext key.

For development / self-hosted, generate a KEK and put it in the configured
environment variable:

```bash
python -c "from src.elicitation.credentials import LocalEnvelopeSecretStore as S; print(S.generate_kek().decode())"
export CLIENT_A_KEK="<the printed key>"
```

Store the KEK in a secret manager, never in the repository. In production,
swap `backend: local_envelope` for a managed-KMS backend; the stored-token
format is unchanged, so existing credentials keep working.

## 4. Create the database schema

Point Alembic at the deployment's database via `ELICITATION_DB_URL` and migrate:

```bash
export ELICITATION_DB_URL="$CLIENT_A_DB_URL"
alembic upgrade head
```

This works against a fresh SQLite file (development) or a fresh Postgres
database (production). To roll back to an empty schema:

```bash
alembic downgrade base
```

Each deployment runs this against its **own** database URL. Two deployments
never share a database.

## 5. Register credentials

```python
from src.elicitation.credentials import CredentialResolver, LocalEnvelopeSecretStore
# (CredentialStore backed by this deployment's DB; in-memory store shown in tests)

secret_store = LocalEnvelopeSecretStore.from_env(cfg.credentials.secret_store.kek_env)
resolver = CredentialResolver(cfg, store, secret_store)

# operator-set deployment key
resolver.store_deployment_key("anthropic", api_key="sk-...")

# a user's BYOK key (allowed only for allowlisted providers)
resolver.store_byok("anthropic", owner_user_id="u1", api_key="sk-...")

# resolve at the point of use (returns a ProviderCredential; .reveal() at the
# HTTP boundary only — the secret never appears in logs or provenance)
cred = resolver.resolve("anthropic", owner_user_id="u1")
```

## 6. Isolation guarantees (verified by tests)

- **Two databases, no cross-visibility.** Rows created in one deployment's
  database are invisible to another (`tests/elicitation/test_schema.py`).
- **Credentials are deployment-scoped.** A resolver only reads rows for its own
  deployment, even if two resolvers share a store
  (`tests/elicitation/test_credentials.py`).
- **Secrets never leak.** A credential's `repr`/`str`, `to_audit_dict()`, and
  any log line redact the key; provenance records the model identity and a
  non-reversible fingerprint only.
- **Migrations round-trip.** `upgrade head` then `downgrade base` returns the
  database to empty (`tests/elicitation/test_migrations.py`).

Run the Layer 0 suite with:

```bash
pytest tests/elicitation -q
```
