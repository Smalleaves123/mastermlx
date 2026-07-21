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


@lru_cache(maxsize=3)
def _load_cpp_backend(backend=None):
    if backend is None:
        backend = get_backend()
    if backend != "auto":
        return None
    try:
        return importlib.import_module("mastermlx.accel._signal_cpp")
    except ImportError:
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

    cpp = _load_cpp_backend(get_backend())
    if cpp is not None and callable(getattr(cpp, "iir_filter_1d", None)):
        return cpp.iir_filter_1d(
            np.ascontiguousarray(x, dtype=np.float64),
            np.ascontiguousarray(b, dtype=np.float64),
            np.ascontiguousarray(a, dtype=np.float64),
        )
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


def frame_signal(x, frame_length, hop_length, pad_end=False):
    """Frame a signal using the C++ kernel when the auto backend is active."""

    x = float_array(x, 1, "x")
    frame_length = int(frame_length)
    hop_length = int(hop_length)
    if frame_length < 1 or hop_length < 1:
        raise ValueError("frame_length and hop_length must be at least 1")
    cpp = _load_cpp_backend(get_backend())
    if cpp is not None and callable(getattr(cpp, "frame_signal", None)):
        return cpp.frame_signal(x, frame_length, hop_length, bool(pad_end))

    if pad_end:
        remainder = (x.size - frame_length) % hop_length
        pad = frame_length - x.size if x.size < frame_length else (0 if remainder == 0 else hop_length - remainder)
        if pad > 0:
            x = np.pad(x, (0, pad))
    if x.size < frame_length:
        return np.empty((0, frame_length), dtype=float)
    n_frames = 1 + (x.size - frame_length) // hop_length
    return np.vstack([x[i * hop_length : i * hop_length + frame_length] for i in range(n_frames)])


def online_cusum(
    x,
    baseline,
    baseline_mean,
    positive,
    negative,
    samples_seen,
    threshold,
    drift,
    baseline_window,
    cooldown_left,
    cooldown,
):
    """Update a CUSUM state and return events plus the updated state."""

    x = float_array(x, 1, "x")
    baseline = np.asarray(baseline, dtype=float).ravel()
    if baseline.size and not np.isfinite(baseline).all():
        raise ValueError("baseline must contain only finite values")
    threshold = float(threshold)
    drift = float(drift)
    baseline_window = int(baseline_window)
    cooldown_left = int(cooldown_left)
    cooldown = int(cooldown)
    if threshold <= 0.0 or baseline_window < 1 or cooldown_left < 0 or cooldown < 0:
        raise ValueError("invalid CUSUM parameters")

    cpp = _load_cpp_backend(get_backend())
    if cpp is not None and callable(getattr(cpp, "online_cusum", None)):
        mean = np.nan if baseline_mean is None else float(baseline_mean)
        return cpp.online_cusum(
            x,
            np.ascontiguousarray(baseline),
            mean,
            float(positive),
            float(negative),
            int(samples_seen),
            threshold,
            drift,
            baseline_window,
            cooldown_left,
            cooldown,
        )

    events = []
    baseline_mean = None if baseline_mean is None else float(baseline_mean)
    positive = float(positive)
    negative = float(negative)
    samples_seen = int(samples_seen)
    for value in x:
        index = samples_seen
        samples_seen += 1
        if baseline_mean is None:
            baseline = np.append(baseline, value)
            if baseline.size >= baseline_window:
                baseline_mean = float(np.mean(baseline))
            continue
        deviation = float(value) - baseline_mean
        positive = max(0.0, positive + deviation - drift)
        negative = min(0.0, negative + deviation + drift)
        if cooldown_left > 0:
            cooldown_left -= 1
            continue
        if positive >= threshold or -negative >= threshold:
            events.append(index)
            positive = 0.0
            negative = 0.0
            cooldown_left = cooldown
    return (
        np.asarray(events, dtype=int),
        baseline,
        baseline_mean,
        positive,
        negative,
        samples_seen,
        cooldown_left,
    )


__all__ = ["frame_signal", "iir_filter_1d", "online_cusum"]
