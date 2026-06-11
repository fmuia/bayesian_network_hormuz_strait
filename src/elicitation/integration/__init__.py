"""Whole-network elicitation integrated with the main dashboard (Plan 4, Layer 4).

Reuses the engine, protocols, agents, and NetworkSpec to run a real (Claude /
optional OpenAI) multi-agent elicitation over the whole network, with effort
controls, then feed the locked result (means + per-CPT kappa) into the existing
inference + credible-interval machinery.
"""

from __future__ import annotations

from .clients import (
    ClaudeClient,
    ElicitationClient,
    OpenAIClient,
    ScriptedClient,
    available_providers,
    make_client,
)
from .framework import (
    EffortConfig,
    ElicitationFramework,
    KNOWN_MODELS,
    REASONING_EFFORTS,
    ModelSpec,
    available_models,
    default_framework,
    list_frameworks,
    load_framework,
    save_framework,
)
from .runner import (
    Agent,
    CallRecord,
    ElicitationRun,
    NodeElicitation,
    build_agents,
    probe_call,
    project_runtime,
    run_elicitation,
)
from .seeds import default_seeds, load_seeds, save_seeds, slug_id
from .store import list_runs, load_run, run_from_dict, run_to_dict, save_run

__all__ = [
    # clients
    "ElicitationClient",
    "ClaudeClient",
    "OpenAIClient",
    "ScriptedClient",
    "available_providers",
    "make_client",
    # framework
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
    "default_seeds",
    "save_seeds",
    "load_seeds",
    "slug_id",
    # runner
    "Agent",
    "CallRecord",
    "NodeElicitation",
    "ElicitationRun",
    "run_elicitation",
    "build_agents",
    "probe_call",
    "project_runtime",
    # store
    "save_run",
    "load_run",
    "list_runs",
    "run_to_dict",
    "run_from_dict",
]
