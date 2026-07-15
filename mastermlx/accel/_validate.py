"""Common boundary checks for optional compiled kernels."""

from __future__ import annotations

import numpy as np


def float_array(value, ndim, name):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if array.ndim != int(ndim):
        raise ValueError(f"{name} must be {ndim}D, got {array.shape}")
    if array.size == 0 or any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def int_arg(value, name, minimum=None):
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result
