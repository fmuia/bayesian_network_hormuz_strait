"""Pack registry: resolve a pack id to its :class:`~packs.base.ScenarioPack`.

The active pack is chosen by the ``SCENARIO_PACK`` environment variable. Packs
are imported lazily (only when first requested) so importing the registry never
drags in every scenario's dependencies, and a broken pack can't break the others.

Default is ``hormuz`` for now; Phase C flips it to ``meridian`` once that pack
exists and is the flagship example.
"""
from __future__ import annotations

import os
from typing import Dict, List

from packs.base import ScenarioPack

DEFAULT_PACK = "hormuz"
_KNOWN: List[str] = ["hormuz", "meridian", "taiwan"]
_CACHE: Dict[str, ScenarioPack] = {}


def _load(pack_id: str) -> ScenarioPack:
    """Import and return a pack's ``PACK`` object (lazy, per-id)."""
    if pack_id == "hormuz":
        from packs.hormuz import PACK
    elif pack_id == "meridian":
        from packs.meridian import PACK
    elif pack_id == "taiwan":
        from packs.taiwan import PACK
    else:
        raise KeyError(
            f"Unknown scenario pack {pack_id!r}; known: {', '.join(_KNOWN)}"
        )
    PACK.validate()
    return PACK


def get_pack(pack_id: str) -> ScenarioPack:
    if pack_id not in _CACHE:
        _CACHE[pack_id] = _load(pack_id)
    return _CACHE[pack_id]


def active_pack_id() -> str:
    """The configured pack id (``SCENARIO_PACK`` env var, else the default)."""
    return os.environ.get("SCENARIO_PACK", DEFAULT_PACK).strip() or DEFAULT_PACK


def get_active() -> ScenarioPack:
    return get_pack(active_pack_id())


def available() -> List[str]:
    """Pack ids known to the registry (whether or not their module exists yet)."""
    return list(_KNOWN)


def _reset_cache() -> None:
    """Test hook: drop the memoised packs (e.g. after changing SCENARIO_PACK)."""
    _CACHE.clear()


__all__ = ["get_pack", "get_active", "active_pack_id", "available", "DEFAULT_PACK"]
