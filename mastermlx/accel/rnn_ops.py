"""Optional Cython and NumPy kernels for recurrent layers."""

from __future__ import annotations

import importlib
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _load_backend():
    try:
        return importlib.import_module("mastermlx.accel._rnn_ops")
    except ImportError:
        return None


def _numpy_simple_rnn(X, W_xh, W_hh, b):
    N, T, _ = X.shape
    H = np.zeros((N, T, W_hh.shape[0]), dtype=float)
    h = np.zeros((N, W_hh.shape[0]), dtype=float)
    for t in range(T):
        h = np.tanh(X[:, t, :] @ W_xh + h @ W_hh + b)
        H[:, t, :] = h
    return H


def simple_rnn_forward(X, W_xh, W_hh, b):
    mod = _load_backend()
    if mod is not None:
        return mod.simple_rnn_forward(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(W_xh, dtype=np.float64),
            np.ascontiguousarray(W_hh, dtype=np.float64),
            np.ascontiguousarray(b, dtype=np.float64),
        )
    return _numpy_simple_rnn(X, W_xh, W_hh, b)


__all__ = ["simple_rnn_forward"]
