from __future__ import annotations

import os
import warnings


_VALID_BACKENDS = {"auto", "numpy", "cython"}
_backend = os.environ.get("MASTERML_BACKEND", "auto").lower()
if _backend not in _VALID_BACKENDS:
    warnings.warn(
        f"invalid MASTERML_BACKEND={_backend!r}; falling back to 'auto'",
        RuntimeWarning,
        stacklevel=2,
    )
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
