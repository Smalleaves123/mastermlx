from __future__ import annotations

import numpy as np


def ensure_1d(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Expected a 1D signal")
    return x


def pad_signal(x, target_length, fill_value=0.0):
    x = ensure_1d(x)
    target_length = int(target_length)
    if target_length < x.size:
        raise ValueError("target_length must be at least len(x)")
    if target_length == x.size:
        return x.copy()
    return np.pad(x, (0, target_length - x.size), constant_values=float(fill_value))


def trim_signal(x, target_length):
    x = ensure_1d(x)
    target_length = int(target_length)
    if target_length < 0:
        raise ValueError("target_length must be non-negative")
    return x[:target_length].copy()


def resample_linear(x, source_rate, target_rate):
    x = ensure_1d(x)
    source_rate = float(source_rate)
    target_rate = float(target_rate)
    if source_rate <= 0.0 or target_rate <= 0.0:
        raise ValueError("sample rates must be positive")
    if x.size == 0:
        return x.copy()
    duration = (x.size - 1) / source_rate
    n_samples = int(round(duration * target_rate)) + 1
    t_old = np.arange(x.size, dtype=float) / source_rate
    t_new = np.linspace(0.0, duration, n_samples)
    return np.interp(t_new, t_old, x)


def chunk_signal(x, chunk_size, step=None, drop_last=False, pad_end=False, fill_value=0.0):
    x = ensure_1d(x)
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    step = chunk_size if step is None else int(step)
    if step < 1:
        raise ValueError("step must be at least 1")
    if x.size == 0:
        return np.empty((0, chunk_size), dtype=float)

    chunks = []
    start = 0
    while start + chunk_size <= x.size:
        chunks.append(x[start : start + chunk_size])
        start += step

    if not drop_last and start < x.size:
        tail = x[start:]
        if pad_end:
            tail = np.pad(tail, (0, chunk_size - tail.size), constant_values=float(fill_value))
            chunks.append(tail)
        elif tail.size > 0:
            chunks.append(tail)

    if not chunks:
        return np.empty((0, chunk_size), dtype=float)

    max_len = max(chunk.size for chunk in chunks)
    out = np.full((len(chunks), max_len), float(fill_value), dtype=float)
    for i, chunk in enumerate(chunks):
        out[i, : chunk.size] = chunk
    return out


def stack_signal_features(features):
    arrays = [np.asarray(feat, dtype=float).ravel() for feat in features]
    if not arrays:
        return np.empty((0, 0), dtype=float)
    widths = {arr.size for arr in arrays}
    if len(widths) != 1:
        raise ValueError("All feature vectors must have the same length")
    return np.vstack(arrays)


__all__ = ["chunk_signal", "ensure_1d", "pad_signal", "resample_linear", "stack_signal_features", "trim_signal"]
