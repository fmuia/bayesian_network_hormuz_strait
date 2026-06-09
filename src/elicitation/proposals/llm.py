"""RAG-augmented CPT proposals (Plan 4, Layer 6.5, optional).

For a CPT being elicited, retrieve the most relevant analog historical events
from the translator audit log and ask an LLM to propose initial values with
span-grounded citations. The proposal commits nothing — an expert reviews,
edits, or rejects it. The proposer is the lowest-autonomy AI mode; humans remain
the experts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..agents.llm_expert import CompletionClient, normalize
from ..protocols.targets import CPTColumnTarget


@dataclass(frozen=True)
class AnalogEvent:
    """A retrieved historical analog with a citation back to its source span."""

    text: str
    citation: str


class Retriever(Protocol):
    """Retrieves analog events for a target (backed by Plan 2's RAG memory)."""

    def retrieve(self, query: str, k: int) -> list[AnalogEvent]: ...


@dataclass(frozen=True)
class CptProposal:
    node: str
    parent_config: tuple[str, ...]
    distribution: tuple[float, ...]
    citations: tuple[str, ...]
    rationale: str
    committed: bool = False  # always False — a proposal commits nothing


def propose_cpt(
    target: CPTColumnTarget,
    retriever: Retriever,
    client: CompletionClient,
    k: int = 5,
) -> CptProposal:
    """Retrieve analogs and ask the model for an initial distribution.

    The returned proposal carries the citations of the analogs that informed it,
    so the reviewing expert can trace every number to a source span.
    """
    analogs = retriever.retrieve(target.describe(), k)
    context = "\n".join(f"- {a.text} [{a.citation}]" for a in analogs)
    prompt = f"{target.describe()}. Relevant analog events:\n{context}"
    raw = client.target_distribution(target.node, prompt, target.states, role=None)
    distribution = normalize(raw)
    target.validate_distribution(distribution)
    return CptProposal(
        node=target.node,
        parent_config=target.parent_config,
        distribution=distribution,
        citations=tuple(a.citation for a in analogs),
        rationale=f"Proposed from {len(analogs)} retrieved analog events.",
        committed=False,
    )


__all__ = ["AnalogEvent", "Retriever", "CptProposal", "propose_cpt"]
