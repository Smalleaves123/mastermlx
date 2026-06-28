from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils.metrics import mean_squared_error


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
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="valid")


def autocorrelation(x, lag=1, demean=True):
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


class ARModel(BaseEstimator):
    """Autoregressive model fitted by least squares."""

    def __init__(self, order=1, fit_intercept=True):
        self.order = int(order)
        self.fit_intercept = bool(fit_intercept)
        self.coef_ = None
        self.intercept_ = 0.0
        self.residuals_ = None
        self.history_ = None

    def _design_matrix(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size == 0:
            raise ValueError("x must be a non-empty 1D array")
        p = self.order
        if p < 1:
            raise ValueError("order must be at least 1")
        if x.size <= p:
            raise ValueError("x must be longer than the model order")
        y = x[p:]
        cols = [x[p - lag - 1 : x.size - lag - 1] for lag in range(p)]
        X = np.column_stack(cols)
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        return X, y

    def fit(self, x, y=None):
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("x must be a 1D array")
        X, target = self._design_matrix(x)
        beta, *_ = np.linalg.lstsq(X, target, rcond=None)
        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta
        fitted = self.predict(x)
        self.residuals_ = target - fitted
        self.history_ = x.copy()
        return self

    def predict(self, x):
        if self.coef_ is None:
            raise RuntimeError("Model has not been fit yet")
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("x must be a 1D array")
        X, _ = self._design_matrix(x)
        pred = X @ np.r_[self.intercept_, self.coef_] if self.fit_intercept else X @ self.coef_
        return pred

    def forecast(self, steps=1, history=None):
        if self.coef_ is None:
            raise RuntimeError("Model has not been fit yet")
        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if history is None:
            if self.history_ is None:
                raise RuntimeError("No training history available")
            history = self.history_
        history = np.asarray(history, dtype=float).ravel()
        if history.size < self.order:
            raise ValueError("history must contain at least order values")
        values = list(history)
        out = []
        for _ in range(steps):
            lags = np.array(values[-self.order :][::-1], dtype=float)
            pred = self.intercept_ + lags @ self.coef_ if self.fit_intercept else lags @ self.coef_
            out.append(float(pred))
            values.append(float(pred))
        return np.asarray(out, dtype=float)

    def score(self, x, y=None):
        x = np.asarray(x, dtype=float)
        pred = self.predict(x)
        target = x[self.order :]
        return -mean_squared_error(target, pred)
