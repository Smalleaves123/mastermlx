"""Optional Cython kernels for time-frequency dynamic programming."""

from __future__ import annotations

import importlib
from functools import lru_cache

import numpy as np

from ..config import get_backend


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


def ridge_path(score, smoothness, max_jump):
    """Return optimal frequency-bin indices for a score map."""

    score = np.ascontiguousarray(score, dtype=float)
    if score.ndim != 2 or 0 in score.shape:
        raise ValueError("score must be a non-empty 2D array")
    smoothness = float(smoothness)
    if smoothness < 0.0:
        raise ValueError("smoothness must be non-negative")
    if max_jump is not None:
        max_jump = int(max_jump)
        if max_jump < 0:
            raise ValueError("max_jump must be non-negative")
    mod = _load_backend(get_backend())
    if mod is not None and callable(getattr(mod, "ridge_path", None)):
        return mod.ridge_path(score, smoothness, -1 if max_jump is None else max_jump)

    n_freqs, n_times = score.shape
    dynamic = np.full((n_freqs, n_times), -np.inf, dtype=float)
    back = np.zeros((n_freqs, n_times), dtype=np.intp)
    dynamic[:, 0] = score[:, 0]
    for t in range(1, n_times):
        for current in range(n_freqs):
            low = 0 if max_jump is None else max(0, current - max_jump)
            high = n_freqs if max_jump is None else min(n_freqs, current + max_jump + 1)
            previous = np.arange(low, high)
            candidate = dynamic[low:high, t - 1] - smoothness * (previous - current) ** 2
            best = int(np.argmax(candidate))
            back[current, t] = previous[best]
            dynamic[current, t] = score[current, t] + candidate[best]
    indices = np.zeros(n_times, dtype=np.intp)
    indices[-1] = int(np.argmax(dynamic[:, -1]))
    for t in range(n_times - 1, 0, -1):
        indices[t - 1] = back[indices[t], t]
    return indices


__all__ = ["ridge_path"]
