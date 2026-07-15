"""Optional Cython kernels for numerical signal-processing loops."""

from __future__ import annotations

import importlib
from functools import lru_cache

import numpy as np

from ..config import get_backend
from ._validate import float_array


@lru_cache(maxsize=3)
def _load_backend(backend=None):
    if backend is None:
        backend = get_backend()
    if backend == "numpy":
        return None
    try:
        return importlib.import_module("mastermlx.accel._signal_ops")
    except ImportError:
        return None


def iir_filter_1d(x, b, a):
    """Apply a real normalized IIR difference equation."""

    x = float_array(x, 1, "x")
    b = float_array(b, 1, "b")
    a = float_array(a, 1, "a")
    if a[0] == 0.0:
        raise ValueError("a[0] must be non-zero")

    mod = _load_backend(get_backend())
    if mod is not None and callable(getattr(mod, "iir_filter_1d", None)):
        return mod.iir_filter_1d(
            np.ascontiguousarray(x, dtype=np.float64),
            np.ascontiguousarray(b, dtype=np.float64),
            np.ascontiguousarray(a, dtype=np.float64),
        )
    y = np.zeros_like(x)
    for n in range(x.size):
        value = 0.0
        for k in range(min(b.size, n + 1)):
            value += b[k] * x[n - k]
        for k in range(1, min(a.size, n + 1)):
            value -= a[k] * y[n - k]
        y[n] = value
    return y


__all__ = ["iir_filter_1d"]
