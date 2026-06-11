"""Per-deployment authentication and authorization.

The RBAC data model (``User``, ``Role``, ``Permission``) lives in
``src.elicitation.db.schema``. This package is the home for login plumbing and
SSO integration hooks, added when the UI layer (Layer 4) needs them. It is
intentionally minimal at Layer 0.
"""

from __future__ import annotations

__all__: list[str] = []
