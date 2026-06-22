"""The pack contract: ``ScenarioPack`` plus the small value types it holds.

A pack bundles the *content* of one scenario. The shared engine
(``src.inference``, ``src.translator``, ``src.viz``, the dashboard) reads a pack
through the ``src.scenario`` seam and never imports a pack module directly, so
adding a scenario is a matter of authoring one folder — no engine edits.

This module is deliberately dependency-light: it imports nothing from ``src`` so
packs can be constructed and validated in isolation. ``build_network`` is held as
a callable (the existing per-scenario builder), so wrapping a scenario is a move,
not a rewrite — the numbers stay byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class Role(str, Enum):
    """A node's structural role, used for colouring/grouping in the views."""

    DRIVER = "driver"        # exogenous cause (root parent of the latent)
    LATENT = "latent"        # the inferred hidden scenario/regime
    EMISSION = "emission"    # observable effect of the latent
    INDICATOR = "indicator"  # observable measurement of a driver
    DECISION = "decision"    # action node in the decision layer
    OTHER = "other"


@dataclass(frozen=True)
class NodeMeta:
    """Per-node presentation metadata (display only — never affects inference)."""

    label: str = ""
    role: Role = Role.OTHER
    group: str = ""
    rationale: str = ""


@dataclass(frozen=True)
class TranslatorProfile:
    """Scenario-specific domain text spliced into the LLM-translator prompts so
    the shared translator stays scenario-blind."""

    domain: str = ""                   # "the Strait of Hormuz"
    scenario_set_descriptor: str = ""  # "Strait-of-Hormuz scenario set (US–Iran tension, …)"
    relevance_descriptor: str = ""     # "the Strait of Hormuz / US-Iran tension / Gulf shipping or energy"
    situation_descriptor: str = ""     # "Strait of Hormuz / US-Iran situation"


@dataclass(frozen=True)
class Presentation:
    """Pack-specific UI/presentation data (display only — never inference).

    Holds everything the shared views need that is scenario-specific: latent-state
    colours/labels, root-driver styling, the per-topology edge-rationale tables,
    the audit node ordering, the model-overview copy, and the heuristic
    fallback-translator keyword map.
    """

    scenario_color: Dict[str, str] = field(default_factory=dict)        # latent state → hex
    scenario_label: Dict[str, str] = field(default_factory=dict)        # latent state → label
    root_driver_style: Dict[str, tuple] = field(default_factory=dict)   # node → (bg, fg)
    root_driver_colors: Dict[str, tuple] = field(default_factory=dict)  # node → (bg, border, observed)
    edge_rationales: Dict[str, list] = field(default_factory=dict)      # topology → [(p, c, reason)]
    edge_omissions: Dict[str, list] = field(default_factory=dict)       # topology → [(p, c, reason)]
    intermediate_nodes: list = field(default_factory=list)              # audit-view ordering
    model_overview: Dict[str, str] = field(default_factory=dict)        # html fragments
    fallback_keyword_map: list = field(default_factory=list)            # [(keywords, node, state)]


@dataclass(frozen=True)
class ScenarioPack:
    """One scenario's content. ``build_network`` and the content constants are the
    same objects the scenario already exports; the engine consumes them verbatim.
    """

    id: str
    title: str
    domain: str
    states: Dict[str, Sequence[str]]
    build_network: Callable[..., Any]
    latent: str = ""                                    # name of the latent node
    edges: List[Tuple[str, str]] = field(default_factory=list)
    edges_latent: List[Tuple[str, str]] = field(default_factory=list)
    narratives: Dict[str, str] = field(default_factory=dict)
    signatures: Dict[str, Any] = field(default_factory=dict)
    node_meta: Dict[str, NodeMeta] = field(default_factory=dict)
    layout: Dict[str, Any] = field(default_factory=dict)
    display_overrides: Dict[str, str] = field(default_factory=dict)  # raw → display name
    node_title_wrap: Dict[str, str] = field(default_factory=dict)    # display name → wrapped
    presentation: "Presentation" = field(default_factory=lambda: Presentation())
    example_headlines: List[Any] = field(default_factory=list)
    # Filled in later phases; optional so Phase A wraps Hormuz without them.
    decision: Optional[Any] = None
    translator_profile: Optional[Any] = None
    elicitation: Optional[Any] = None

    def validate(self) -> None:
        """Cheap structural checks (not the numeric CPT validation — that lives in
        ``NetworkSpec.validate`` / ``check_model``). Raises ``ValueError`` on a
        malformed pack so a typo surfaces at load time, not mid-demo."""
        if not self.states:
            raise ValueError(f"pack {self.id!r}: empty state vocabulary")
        known = set(self.states)
        if self.latent and self.latent not in known:
            raise ValueError(f"pack {self.id!r}: latent {self.latent!r} not in states")
        for label, edge_set in (("edges", self.edges), ("edges_latent", self.edges_latent)):
            for a, b in edge_set:
                if a not in known or b not in known:
                    raise ValueError(
                        f"pack {self.id!r}: {label} references unknown node in {(a, b)!r}"
                    )
        # node_meta is node-keyed (validated); layout is an opaque, engine-specific
        # blob (e.g. topology-keyed) so it is intentionally not key-checked here.
        for node in self.node_meta:
            if node not in known:
                raise ValueError(
                    f"pack {self.id!r}: node_meta has unknown node {node!r}"
                )
        if self.latent and self.narratives:
            latent_states = set(self.states[self.latent])
            extra = set(self.narratives) - latent_states
            if extra:
                raise ValueError(
                    f"pack {self.id!r}: narratives for non-latent-states {sorted(extra)!r}"
                )


__all__ = ["Role", "NodeMeta", "Presentation", "TranslatorProfile", "ScenarioPack"]
