"""Export/import between the elicitation store and the declarative NetworkSpec."""

from __future__ import annotations

from .network_spec import (
    cpts_to_network_spec,
    network_spec_to_cpts,
    spec_from_dict,
    spec_to_dict,
)

__all__ = [
    "cpts_to_network_spec",
    "network_spec_to_cpts",
    "spec_to_dict",
    "spec_from_dict",
]
