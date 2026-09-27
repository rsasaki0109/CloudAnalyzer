"""Optional acceleration by the Rust core (``pip install "cloudanalyzer[fast]"``).

When the ``cloudanalyzer_core`` extension is installed, hot paths use it; set
``CA_DISABLE_RUST_CORE=1`` to force the pure Open3D/NumPy implementations.
"""

from __future__ import annotations

import os
from functools import lru_cache
from types import ModuleType


@lru_cache(maxsize=1)
def _load() -> ModuleType | None:
    if os.environ.get("CA_DISABLE_RUST_CORE", "").strip() not in {"", "0"}:
        return None
    try:
        import cloudanalyzer_core
    except ImportError:
        return None
    return cloudanalyzer_core


def core() -> ModuleType | None:
    """The ``cloudanalyzer_core`` module, or ``None`` when unavailable or disabled."""
    return _load()
