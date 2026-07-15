from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..config import get_backend
from ..utils.metrics import mean_squared_error, r2_score

try:
    from ._time_series_ops import (
        autocorrelation_1d as _cy_autocorrelation_1d,
        autocorrelation_function_1d as _cy_autocorrelation_function_1d,
        cusum_change_points_1d as _cy_cusum_change_points_1d,
        exponential_smoothing_1d as _cy_exponential_smoothing_1d,
        rolling_mean_1d as _cy_rolling_mean_1d,
    )
except ImportError:  # pragma: no cover - fallback when Cython extensions are unavailable
    _cy_autocorrelation_1d = None
    _cy_autocorrelation_function_1d = None
    _cy_cusum_change_points_1d = None
    _cy_exponential_smoothing_1d = None
    _cy_rolling_mean_1d = None


def _as_1d_series(x, name="x"):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    return x


def _normalize_steps(steps, name="steps"):
    if steps is None:
        return []
    if isinstance(steps, (list, tuple)):
        if not steps:
            raise ValueError(f"{name} must be non-empty")
        first = steps[0]
        if isinstance(first, tuple) and len(first) == 2:
            return list(steps)
        return [(f"{name[:-1] if name.endswith('s') else name}_{idx}", step) for idx, step in enumerate(steps)]
    if hasattr(steps, "fit") and hasattr(steps, "transform"):
        return [(name[:-1] if name.endswith("s") else name, steps)]
    raise TypeError(f"{name} must be None, a transformer, a pipeline, or a list of steps")


def lagged_matrix(x, lags=1, horizon=1):
    x = _as_1d_series(x)
    lags = int(lags)
    horizon = int(horizon)
    if lags < 1:
        raise ValueError("lags must be at least 1")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    n_samples = x.size - lags - horizon + 1
    if n_samples < 1:
        raise ValueError("x is too short for the requested lags and horizon")
    X = np.empty((n_samples, lags), dtype=float)
    y = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        start = i
        stop = i + lags
        X[i] = x[start:stop]
        y[i] = x[stop + horizon - 1]
    return X, y


class LaggedTimeSeriesTransformer(BaseTransformer):
    """Convert a 1D series into a supervised lag matrix."""

    def __init__(self, lags=1, horizon=1):
        self.lags = int(lags)
        self.horizon = int(horizon)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X, _ = lagged_matrix(X, lags=self.lags, horizon=self.horizon)
        return X

    def transform_target(self, X):
        _, y = lagged_matrix(X, lags=self.lags, horizon=self.horizon)
        return y


def difference(x, periods=1, order=1):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    periods = int(periods)
    order = int(order)
    if periods < 1 or order < 1:
        raise ValueError("periods and order must be at least 1")
    out = x
    for _ in range(order):
        if out.size <= periods:
            return np.empty(0, dtype=float)
        out = out[periods:] - out[:-periods]
    return out


def rolling_mean(x, window):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    window = int(window)
    if window < 1:
        raise ValueError("window must be at least 1")
    if window > x.size:
        raise ValueError("window cannot exceed the series length")
    if get_backend() != "numpy" and _cy_rolling_mean_1d is not None:
        return _cy_rolling_mean_1d(x, window)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="valid")


def autocorrelation(x, lag=1, demean=True):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    lag = int(lag)
    if lag < 0:
        raise ValueError("lag must be non-negative")
    if get_backend() != "numpy" and _cy_autocorrelation_1d is not None:
        return float(_cy_autocorrelation_1d(x, lag, demean=demean))
    if lag == 0:
        return 1.0
    if lag >= x.size:
        return 0.0
    centered = x - np.mean(x) if demean else x
    denom = float(np.dot(centered, centered))
    if denom == 0.0:
        return 0.0
    num = float(np.dot(centered[:-lag], centered[lag:]))
    return num / denom


def autocorrelation_function(x, max_lag):
    max_lag = int(max_lag)
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    if get_backend() != "numpy" and _cy_autocorrelation_function_1d is not None:
        return np.asarray(_cy_autocorrelation_function_1d(np.asarray(x, dtype=float), max_lag), dtype=float)
    return np.asarray([autocorrelation(x, lag=lag) for lag in range(max_lag + 1)], dtype=float)


def partial_autocorrelation(x, lag=1):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    lag = int(lag)
    if lag < 0:
        raise ValueError("lag must be non-negative")
    if lag == 0:
        return 1.0
    if lag >= x.size:
        return 0.0
    acf = autocorrelation_function(x, lag)
    toeplitz = np.empty((lag, lag), dtype=float)
    for i in range(lag):
        for j in range(lag):
            toeplitz[i, j] = acf[abs(i - j)]
    rhs = acf[1 : lag + 1]
    try:
        coeffs = np.linalg.solve(toeplitz, rhs)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.pinv(toeplitz) @ rhs
    return float(coeffs[-1])


def exponential_smoothing(x, alpha=0.5):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    alpha = float(alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    if get_backend() != "numpy" and _cy_exponential_smoothing_1d is not None:
        return _cy_exponential_smoothing_1d(x, alpha)
    smoothed = np.empty_like(x, dtype=float)
    smoothed[0] = x[0]
    for i in range(1, x.size):
        smoothed[i] = alpha * x[i] + (1.0 - alpha) * smoothed[i - 1]
    return smoothed


def dtw_path(x, y, window=None):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        raise ValueError("x and y must be non-empty")
    # Try C++ accelerated path
    try:
        if get_backend() == "numpy":
            raise ImportError
        from ..accel._dtw import dtw_path as _cpp_dtw
        w = int(window) if window is not None else max(x.size, y.size)
        path_arr, dist = _cpp_dtw(x, y, w)
        return [(int(p[0]), int(p[1])) for p in path_arr], float(dist)
    except ImportError:
        pass
    n, m = x.size, y.size
    if window is None:
        window = max(n, m)
    window = int(window)
    if window < abs(n - m):
        window = abs(n - m)

    inf = np.inf
    dp = np.full((n + 1, m + 1), inf, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_stop = min(m, i + window)
        for j in range(j_start, j_stop + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        steps = np.array([dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]], dtype=float)
        move = int(np.argmin(steps))
        if move == 0:
            i -= 1
        elif move == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    path.reverse()
    return path, float(dp[n, m])


def dtw_distance(x, y, window=None):
    return dtw_path(x, y, window=window)[1]


def cusum_change_points(x, threshold=5.0, drift=0.0, direction="both"):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    threshold = float(threshold)
    drift = float(drift)
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if direction not in {"positive", "negative", "both"}:
        raise ValueError("direction must be one of: positive, negative, both")

    if get_backend() != "numpy" and _cy_cusum_change_points_1d is not None:
        direction_code = 0 if direction == "both" else 1 if direction == "positive" else 2
        return _cy_cusum_change_points_1d(x, threshold, drift, direction_code)

    mean = np.mean(x)
    pos = 0.0
    neg = 0.0
    cps = []
    for i, value in enumerate(x):
        dev = value - mean - drift
        pos = max(0.0, pos + dev)
        neg = min(0.0, neg + dev)
        if direction in {"positive", "both"} and pos > threshold:
            cps.append(i)
            pos = 0.0
            neg = 0.0
        elif direction in {"negative", "both"} and -neg > threshold:
            cps.append(i)
            pos = 0.0
            neg = 0.0
    return cps


def _score(y, p, scoring):
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if y.shape[0] != p.shape[0]:
        raise ValueError("y and predictions must have the same length")
    if scoring is None or scoring == "neg_mean_squared_error":
        return -mean_squared_error(y, p)
    if scoring == "r2":
        return r2_score(y, p)
    if callable(scoring):
        return float(scoring(y, p))
    raise ValueError("scoring must be None, 'neg_mean_squared_error', 'r2', or a callable")




__all__ = [
    "LaggedTimeSeriesTransformer",
    "autocorrelation",
    "autocorrelation_function",
    "cusum_change_points",
    "difference",
    "dtw_distance",
    "dtw_path",
    "exponential_smoothing",
    "lagged_matrix",
    "partial_autocorrelation",
    "rolling_mean",
]
