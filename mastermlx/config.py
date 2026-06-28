from __future__ import annotations

import os


_VALID_BACKENDS = {"auto", "numpy", "cython"}
_backend = os.environ.get("MASTERML_BACKEND", "auto").lower()
if _backend not in _VALID_BACKENDS:
    _backend = "auto"


def set_backend(name: str):
    """Set the preferred compute backend."""
    global _backend
    name = str(name).lower()
    if name not in _VALID_BACKENDS:
        raise ValueError("backend must be one of: auto, numpy, cython")
    _backend = name


def get_backend():
    """Return the preferred compute backend."""
    return _backend
