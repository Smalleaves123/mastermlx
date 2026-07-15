"""Optional Cython and NumPy kernels for recurrent layers."""

from __future__ import annotations

import importlib
from functools import lru_cache

import numpy as np

from ..config import get_backend
from ._validate import float_array, int_arg

AUTO_MIN_WORK = 1 << 18


def _use_compiled(X, units, module):
    if module is None or get_backend() == "numpy":
        return False
    if get_backend() == "cython":
        return True
    work = int(X.shape[0]) * int(X.shape[1]) * int(X.shape[2]) * int(units)
    return work >= AUTO_MIN_WORK


@lru_cache(maxsize=3)
def _load_backend(backend=None):
    if backend is None:
        backend = get_backend()
    if backend == "numpy":
        return None
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


def _numpy_lstm(X, W, U, b, units):
    N, T, _ = X.shape
    H = np.zeros((N, T, units), dtype=float)
    C = np.zeros((N, T, units), dtype=float)
    gates = np.zeros((N, T, 4 * units), dtype=float)
    h = np.zeros((N, units), dtype=float)
    c = np.zeros((N, units), dtype=float)
    for t in range(T):
        value = X[:, t, :] @ W + h @ U + b
        f = 1.0 / (1.0 + np.exp(-value[:, :units]))
        i = 1.0 / (1.0 + np.exp(-value[:, units:2 * units]))
        g = np.tanh(value[:, 2 * units:3 * units])
        o = 1.0 / (1.0 + np.exp(-value[:, 3 * units:]))
        c = f * c + i * g
        h = o * np.tanh(c)
        H[:, t, :] = h
        C[:, t, :] = c
        gates[:, t, :] = np.concatenate([f, i, g, o], axis=1)
    return H, C, gates


def _numpy_gru(X, W_zr, W_h, U_zr, U_h, b_zr, b_h, units):
    N, T, _ = X.shape
    H = np.zeros((N, T, units), dtype=float)
    gates = np.zeros((N, T, 3 * units), dtype=float)
    h = np.zeros((N, units), dtype=float)
    for t in range(T):
        value = X[:, t, :] @ W_zr + h @ U_zr + b_zr
        z = 1.0 / (1.0 + np.exp(-value[:, :units]))
        r = 1.0 / (1.0 + np.exp(-value[:, units:]))
        h_tilde = np.tanh(X[:, t, :] @ W_h + (r * h) @ U_h + b_h)
        h = (1.0 - z) * h + z * h_tilde
        H[:, t, :] = h
        gates[:, t, :] = np.concatenate([z, r, h_tilde], axis=1)
    return H, gates


def simple_rnn_forward(X, W_xh, W_hh, b):
    X = float_array(X, 3, "X")
    W_xh = float_array(W_xh, 2, "W_xh")
    W_hh = float_array(W_hh, 2, "W_hh")
    b = float_array(b, 1, "b")
    if W_xh.shape[0] != X.shape[2] or W_hh.shape[0] != W_hh.shape[1]:
        raise ValueError("RNN parameter shapes are inconsistent")
    if W_xh.shape[1] != W_hh.shape[0] or b.shape != (W_hh.shape[0],):
        raise ValueError("RNN parameter shapes are inconsistent")
    mod = _load_backend(get_backend())
    if _use_compiled(X, W_hh.shape[0], mod):
        return mod.simple_rnn_forward(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(W_xh, dtype=np.float64),
            np.ascontiguousarray(W_hh, dtype=np.float64),
            np.ascontiguousarray(b, dtype=np.float64),
        )
    return _numpy_simple_rnn(X, W_xh, W_hh, b)


def lstm_forward(X, W, U, b, units):
    X = float_array(X, 3, "X")
    W = float_array(W, 2, "W")
    U = float_array(U, 2, "U")
    b = float_array(b, 1, "b")
    units = int_arg(units, "units", 1)
    if W.shape != (X.shape[2], 4 * units) or U.shape != (units, 4 * units) or b.shape != (4 * units,):
        raise ValueError("LSTM parameter shapes are inconsistent")
    mod = _load_backend(get_backend())
    if _use_compiled(X, units, mod) and callable(getattr(mod, "lstm_forward", None)):
        return mod.lstm_forward(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(W, dtype=np.float64),
            np.ascontiguousarray(U, dtype=np.float64),
            np.ascontiguousarray(b, dtype=np.float64),
            int(units),
        )
    return _numpy_lstm(X, W, U, b, int(units))


def gru_forward(X, W_zr, W_h, U_zr, U_h, b_zr, b_h, units):
    X = float_array(X, 3, "X")
    W_zr = float_array(W_zr, 2, "W_zr")
    W_h = float_array(W_h, 2, "W_h")
    U_zr = float_array(U_zr, 2, "U_zr")
    U_h = float_array(U_h, 2, "U_h")
    b_zr = float_array(b_zr, 1, "b_zr")
    b_h = float_array(b_h, 1, "b_h")
    units = int_arg(units, "units", 1)
    if (
        W_zr.shape != (X.shape[2], 2 * units)
        or W_h.shape != (X.shape[2], units)
        or U_zr.shape != (units, 2 * units)
        or U_h.shape != (units, units)
        or b_zr.shape != (2 * units,)
        or b_h.shape != (units,)
    ):
        raise ValueError("GRU parameter shapes are inconsistent")
    mod = _load_backend(get_backend())
    if _use_compiled(X, units, mod) and callable(getattr(mod, "gru_forward", None)):
        return mod.gru_forward(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(W_zr, dtype=np.float64),
            np.ascontiguousarray(W_h, dtype=np.float64),
            np.ascontiguousarray(U_zr, dtype=np.float64),
            np.ascontiguousarray(U_h, dtype=np.float64),
            np.ascontiguousarray(b_zr, dtype=np.float64),
            np.ascontiguousarray(b_h, dtype=np.float64),
            int(units),
        )
    return _numpy_gru(X, W_zr, W_h, U_zr, U_h, b_zr, b_h, int(units))


__all__ = ["gru_forward", "lstm_forward", "simple_rnn_forward"]
