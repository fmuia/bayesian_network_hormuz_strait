"""Tests for per-deployment configuration loading and BYOK policy."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.elicitation.config import CredentialSource, DeploymentConfig, load_deployment_config


def test_defaults_are_safe() -> None:
    """A minimal config has BYOK disabled and a sane resolution order."""
    cfg = DeploymentConfig.model_validate({"name": "client_a"})
    assert cfg.credentials.byok.enabled is False
    assert cfg.credentials.byok.provider_allowlist == []
    assert cfg.credentials.resolution_order[0] is CredentialSource.BYOK
    assert cfg.database.url.startswith("sqlite")


def test_byok_allowlist_is_normalised_and_enforced() -> None:
    cfg = DeploymentConfig.model_validate(
        {
            "name": "client_a",
            "credentials": {
                "byok": {"enabled": True, "provider_allowlist": [" Anthropic ", "OpenAI"]}
            },
        }
    )
    assert cfg.credentials.byok.provider_allowlist == ["anthropic", "openai"]
    assert cfg.credentials.byok.allows("anthropic") is True
    assert cfg.credentials.byok.allows("ANTHROPIC") is True
    assert cfg.credentials.byok.allows("cohere") is False


def test_byok_disabled_allows_nothing() -> None:
    cfg = DeploymentConfig.model_validate(
        {"name": "x", "credentials": {"byok": {"enabled": False, "provider_allowlist": ["anthropic"]}}}
    )
    assert cfg.credentials.byok.allows("anthropic") is False


def test_unknown_keys_rejected() -> None:
    with pytest.raises(Exception):
        DeploymentConfig.model_validate({"name": "x", "bogus_field": 1})


def test_empty_name_rejected() -> None:
    with pytest.raises(Exception):
        DeploymentConfig.model_validate({"name": "   "})


def test_yaml_loader_with_env_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_DB_URL", "sqlite:////tmp/client_a.db")
    cfg_file = tmp_path / "deployment.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """
            name: client_a
            topology: latent_regime
            database:
              url: ${TEST_DB_URL}
            credentials:
              byok:
                enabled: true
                provider_allowlist: [anthropic, openai]
            """
        )
    )
    cfg = load_deployment_config(cfg_file)
    assert cfg.name == "client_a"
    assert cfg.database.url == "sqlite:////tmp/client_a.db"
    assert cfg.credentials.byok.allows("openai") is True


def test_yaml_loader_undefined_env_raises(tmp_path: Path) -> None:
    cfg_file = tmp_path / "deployment.yaml"
    cfg_file.write_text("name: x\ndatabase:\n  url: ${DEFINITELY_NOT_SET_VAR}\n")
    with pytest.raises(KeyError):
        load_deployment_config(cfg_file)
