"""Correlation and correlation-peak analysis for sampled signals."""

from __future__ import annotations

import numpy as np


def _signal(x, name="x"):
    arr = np.asarray(x)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    return arr.astype(complex if np.iscomplexobj(arr) else float, copy=False)


def _slice_lags(values, lags, max_lag, two_sided):
    if max_lag is None:
        max_lag = int(np.max(np.abs(lags)))
    max_lag = int(max_lag)
    if max_lag < 0 or max_lag > int(np.max(np.abs(lags))):
        raise ValueError("max_lag is outside the available correlation lags")
    if two_sided:
        mask = np.abs(lags) <= max_lag
    else:
        mask = (lags >= 0) & (lags <= max_lag)
    return lags[mask], values[mask]


def signal_autocorrelation(x, max_lag=None, demean=True, normalize=True, two_sided=False):
    """Return autocorrelation lags and values.

    Positive lags use the convention ``R[k] = sum(conj(x[n]) x[n+k])``.
    """

    x = _signal(x)
    if demean:
        x = x - np.mean(x)
    values = np.correlate(x, x, mode="full")
    lags = np.arange(-(x.size - 1), x.size)
    if normalize:
        scale = float(np.real(values[x.size - 1]))
        if scale > 0.0:
            values = values / scale
    return _slice_lags(values, lags, max_lag, two_sided)


def signal_cross_correlation(
    x,
    y,
    max_lag=None,
    demean=True,
    normalize=True,
    two_sided=True,
):
    """Return cross-correlation lags and values.

    A positive lag means that ``y`` is delayed relative to ``x``.  Complex
    signals use the usual conjugate cross-correlation convention.
    """

    x = _signal(x, "x")
    y = _signal(y, "y")
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if demean:
        x = x - np.mean(x)
        y = y - np.mean(y)
    values = np.correlate(y, x, mode="full")
    lags = np.arange(-(x.size - 1), x.size)
    if normalize:
        scale = np.sqrt(np.sum(np.abs(x) ** 2) * np.sum(np.abs(y) ** 2))
        if scale > 0.0:
            values = values / scale
    return _slice_lags(values, lags, max_lag, two_sided)


def correlation_peaks(
    values,
    lags=None,
    n_peaks=5,
    min_distance=1,
    threshold=None,
    positive_only=False,
):
    """Return prominent local correlation peaks as ``(lag, value)`` rows."""

    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1D array")
    if lags is None:
        lags = np.arange(values.size)
    lags = np.asarray(lags)
    if lags.ndim != 1 or lags.size != values.size:
        raise ValueError("lags must match values")
    n_peaks = int(n_peaks)
    min_distance = int(min_distance)
    if n_peaks < 0:
        raise ValueError("n_peaks must be non-negative")
    if min_distance < 1:
        raise ValueError("min_distance must be positive")
    if n_peaks == 0:
        return np.empty((0, 2), dtype=float)

    score = np.real(values) if np.iscomplexobj(values) else values.astype(float, copy=False)
    candidates = []
    for idx in range(values.size):
        left = score[idx - 1] if idx > 0 else -np.inf
        right = score[idx + 1] if idx + 1 < values.size else -np.inf
        if score[idx] >= left and score[idx] > right:
            if positive_only and score[idx] <= 0.0:
                continue
            if threshold is not None and score[idx] < float(threshold):
                continue
            candidates.append(idx)
    candidates.sort(key=lambda idx: score[idx], reverse=True)
    selected: list[int] = []
    for idx in candidates:
        if all(abs(idx - previous) >= min_distance for previous in selected):
            selected.append(idx)
            if len(selected) == n_peaks:
                break
    selected.sort(key=lambda idx: lags[idx])
    return np.column_stack([lags[selected], values[selected]]) if selected else np.empty((0, 2), dtype=values.dtype)


def autocorrelation_peaks(x, max_lag=None, n_peaks=5, min_distance=1, exclude_zero=True):
    """Extract peaks from the one- or two-sided autocorrelation."""

    lags, values = signal_autocorrelation(x, max_lag=max_lag, two_sided=False)
    if exclude_zero:
        mask = lags != 0
        lags, values = lags[mask], values[mask]
    if values.size == 0:
        return np.empty((0, 2), dtype=float)
    return correlation_peaks(values, lags, n_peaks=n_peaks, min_distance=min_distance, positive_only=True)


def cross_correlation_peaks(x, y, max_lag=None, n_peaks=5, min_distance=1, positive_only=False):
    """Extract peaks from the cross-correlation of two signals."""

    lags, values = signal_cross_correlation(x, y, max_lag=max_lag, two_sided=True)
    return correlation_peaks(
        values,
        lags,
        n_peaks=n_peaks,
        min_distance=min_distance,
        positive_only=positive_only,
    )


__all__ = [
    "autocorrelation_peaks",
    "correlation_peaks",
    "cross_correlation_peaks",
    "signal_autocorrelation",
    "signal_cross_correlation",
]
