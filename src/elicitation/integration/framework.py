"""Elicitation frameworks: a saveable run configuration.

A framework records *which models* to draw agents from (only models that have
SDK/key wiring are offered), *how many agents* to recruit, the effort limits
that keep a whole-network run inside a time budget, and the calibration seeds.
Roles are NOT fixed here — the LLM chooses node-appropriate roles at run time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..protocols.base import SeedQuestion
from .clients import ClaudeClient, OpenAIClient
from .seeds import default_seeds


@dataclass(frozen=True)
class ModelSpec:
    """A base model the panel can draw agents from."""

    provider: str       # "claude-code" | "openai"
    model: str
    label: str

    def available(self) -> bool:
        if self.provider == "claude-code":
            return ClaudeClient.available()
        if self.provider == "openai":
            return OpenAIClient.available()
        return False


# Models with SDK/key wiring. ``available()`` reflects what can actually run now.
KNOWN_MODELS: list[ModelSpec] = [
    ModelSpec("claude-code", "claude-sonnet-4-5", "Claude Sonnet 4.5 (Max sub)"),
    ModelSpec("claude-code", "claude-opus-4-1", "Claude Opus 4.1 (Max sub)"),
    ModelSpec("openai", "gpt-4o-mini", "OpenAI GPT-4o-mini (API key)"),
    ModelSpec("openai", "gpt-4o", "OpenAI GPT-4o (API key)"),
]


def available_models() -> list[ModelSpec]:
    """The subset of known models that can be called right now."""
    return [m for m in KNOWN_MODELS if m.available()]


REASONING_EFFORTS = ("brief", "balanced", "thorough")


@dataclass
class EffortConfig:
    """Performance controls. A selected node is never dropped for being slow.

    ``reasoning_effort`` sets how much an agent deliberates on its first attempt.
    ``time_limit_s`` is a soft budget: if the agent is still going when it elapses,
    it is re-asked to conclude immediately (one time-boxed escalation, not a retry
    loop). ``hard_timeout_s`` is a final guard for a genuinely hung call.
    """

    n_seeds: int = 0                   # 0 = use all authored seeds
    concurrency: int = 4               # max LLM calls in flight at once (across all nodes AND agents)
    reasoning_effort: str = "balanced"  # brief | balanced | thorough
    time_limit_s: float = 45.0         # soft budget; on exceed the agent is asked to conclude now
    hard_timeout_s: float = 180.0      # final wall-clock guard for a hung call


@dataclass
class ElicitationFramework:
    name: str
    models: list[ModelSpec] = field(default_factory=lambda: [KNOWN_MODELS[0]])
    n_agents: int = 3
    nodes: list[str] = field(default_factory=list)  # which CPTs to elicit; empty = whole network
    effort: EffortConfig = field(default_factory=EffortConfig)
    seeds: list[SeedQuestion] = field(default_factory=default_seeds)

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "models": [asdict(m) for m in self.models],
            "n_agents": self.n_agents,
            "nodes": list(self.nodes),
            "effort": asdict(self.effort),
            "seeds": [
                {"id": s.id, "text": s.text, "realization": s.realization, "unit": s.unit}
                for s in self.seeds
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ElicitationFramework":
        return cls(
            name=data["name"],
            models=[ModelSpec(**m) for m in data["models"]],
            n_agents=int(data.get("n_agents", 3)),
            nodes=list(data.get("nodes", [])),
            effort=EffortConfig(**data.get("effort", {})),
            seeds=[SeedQuestion(**s) for s in data.get("seeds", [])] or default_seeds(),
        )


def default_framework() -> ElicitationFramework:
    """A sensible default: Claude-only, 3 agents, modest effort cap."""
    models = available_models() or [KNOWN_MODELS[0]]
    claude = [m for m in models if m.provider == "claude-code"] or [models[0]]
    return ElicitationFramework(name="default", models=claude[:1], n_agents=3)


# -- framework library (JSON files) ------------------------------------------ #


def save_framework(framework: ElicitationFramework, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{framework.name}.json"
    path.write_text(json.dumps(framework.to_dict(), indent=2))
    return path


def load_framework(name: str, directory: str | Path) -> ElicitationFramework:
    path = Path(directory) / f"{name}.json"
    return ElicitationFramework.from_dict(json.loads(path.read_text()))


def list_frameworks(directory: str | Path) -> list[str]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.json"))


__all__ = [
    "ModelSpec",
    "EffortConfig",
    "ElicitationFramework",
    "REASONING_EFFORTS",
    "KNOWN_MODELS",
    "available_models",
    "default_framework",
    "save_framework",
    "load_framework",
    "list_frameworks",
]
