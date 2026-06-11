"""Alembic migration environment for the elicitation platform.

The target metadata is ``src.elicitation.db.schema.Base.metadata``. The database
URL is taken from the ``ELICITATION_DB_URL`` environment variable when set (so a
migration can target any deployment's isolated database), otherwise from
``sqlalchemy.url`` in ``alembic.ini``. ``render_as_batch`` is enabled so SQLite
deployments can run ALTER-style migrations.
"""

from logging.config import fileConfig
import os

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

from src.elicitation.db.schema import Base

# Alembic Config object, providing access to the .ini file values.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Allow any deployment's database to be targeted via the environment.
_env_url = os.environ.get("ELICITATION_DB_URL")
if _env_url:
    config.set_main_option("sqlalchemy.url", _env_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (URL only, no DBAPI)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
