"""Optional Cython and NumPy kernels for 1D convolution."""

from __future__ import annotations

import importlib
from functools import lru_cache

import numpy as np

from ..config import get_backend
from ._validate import float_array, int_arg


@lru_cache(maxsize=3)
def _load_backend(backend=None):
    if backend is None:
        backend = get_backend()
    if backend == "numpy":
        return None
    try:
        return importlib.import_module("mastermlx.accel._conv1d_ops")
    except ImportError:
        return None


def _numpy_im2col(X, kernel_size, stride=1, pad=0):
    X = np.asarray(X, dtype=float)
    N, T, C = X.shape
    X_pad = np.pad(X, ((0, 0), (pad, pad), (0, 0))) if pad else X
    out_t = (T + 2 * pad - kernel_size) // stride + 1
    windows = np.lib.stride_tricks.sliding_window_view(X_pad, kernel_size, axis=1)
    windows = windows[:, ::stride, :, :].transpose(0, 1, 3, 2)
    return np.ascontiguousarray(windows.reshape(N * out_t, kernel_size * C)), out_t


def _numpy_col2im(cols, shape, kernel_size, stride=1, pad=0):
    N, T, C = shape
    out_t = (T + 2 * pad - kernel_size) // stride + 1
    X_pad = np.zeros((N, T + 2 * pad, C), dtype=float)
    values = np.asarray(cols, dtype=float).reshape(N, out_t, kernel_size, C)
    batch_idx = np.arange(N)[:, None, None, None]
    time_idx = (np.arange(out_t) * stride)[None, :, None, None] + np.arange(kernel_size)[None, None, :, None]
    channel_idx = np.arange(C)[None, None, None, :]
    np.add.at(X_pad, (batch_idx, time_idx, channel_idx), values)
    return X_pad[:, pad:-pad, :] if pad else X_pad


def im2col1d(X, kernel_size, stride=1, pad=0):
    X = float_array(X, 3, "X")
    kernel_size = int_arg(kernel_size, "kernel_size", 1)
    stride = int_arg(stride, "stride", 1)
    pad = int_arg(pad, "pad", 0)
    if X.shape[1] + 2 * pad < kernel_size:
        raise ValueError("kernel_size is larger than the padded sequence")
    mod = _load_backend(get_backend())
    if mod is not None:
        return mod.im2col1d(np.ascontiguousarray(X, dtype=np.float64), kernel_size, stride, pad)
    return _numpy_im2col(X, kernel_size, stride, pad)


def col2im1d(cols, shape, kernel_size, stride=1, pad=0):
    cols = float_array(cols, 2, "cols")
    if len(shape) != 3 or any(int(size) < 1 for size in shape):
        raise ValueError("shape must contain three positive dimensions")
    kernel_size = int_arg(kernel_size, "kernel_size", 1)
    stride = int_arg(stride, "stride", 1)
    pad = int_arg(pad, "pad", 0)
    output = (int(shape[1]) + 2 * pad - kernel_size) // stride + 1
    if output < 1 or cols.shape[0] != int(shape[0]) * output:
        raise ValueError("cols shape is inconsistent with shape and kernel parameters")
    mod = _load_backend(get_backend())
    if mod is not None:
        return mod.col2im1d(np.ascontiguousarray(cols, dtype=np.float64), shape, kernel_size, stride, pad)
    return _numpy_col2im(cols, shape, kernel_size, stride, pad)


__all__ = ["col2im1d", "im2col1d"]
