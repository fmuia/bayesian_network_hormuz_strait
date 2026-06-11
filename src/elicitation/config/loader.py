"""Load a per-deployment configuration from YAML.

``${ENV_VAR}`` references in string values are expanded from the environment,
so secrets (e.g. a database URL with a password) never need to live in the
YAML file itself. The result is a validated :class:`DeploymentConfig`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .models import DeploymentConfig

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` in strings; leave other types untouched."""
    if isinstance(value, str):
        def _sub(match: re.Match[str]) -> str:
            var = match.group(1)
            replacement = os.environ.get(var)
            if replacement is None:
                raise KeyError(
                    f"config references undefined environment variable ${{{var}}}"
                )
            return replacement

        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_deployment_config(path: str | Path) -> DeploymentConfig:
    """Read, env-expand, and validate a deployment config YAML file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"deployment config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"deployment config must be a mapping, got {type(raw).__name__}")
    return DeploymentConfig.model_validate(_expand_env(raw))
