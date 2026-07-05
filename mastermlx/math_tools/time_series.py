from __future__ import annotations

import numpy as np

from ..base import BaseEstimator, BaseTransformer
from ..utils.estimator import clone
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
    if _cy_rolling_mean_1d is not None:
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
    if _cy_autocorrelation_1d is not None:
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
    if _cy_autocorrelation_function_1d is not None:
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
    if _cy_exponential_smoothing_1d is not None:
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

    if _cy_cusum_change_points_1d is not None:
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


class TimeSeriesPipeline(BaseEstimator):
    """Fit a model on lagged time-series windows with optional preprocessing."""

    def __init__(self, model, lags=12, horizon=1, preprocessing=None):
        self.model = model
        self.lags = int(lags)
        self.horizon = int(horizon)
        self.preprocessing = preprocessing
        self.pipeline_ = None
        self.model_ = None
        self.feature_transformer_ = None
        self.history_ = None
        self.training_features_ = None
        self.training_target_ = None
        self.mode_ = None

    def _resolve_preprocessing(self):
        return _normalize_steps(self.preprocessing, name="preprocessing")

    def _fit_transform_preprocessing(self, X, y=None):
        steps = self._resolve_preprocessing()
        Xt = np.asarray(X, dtype=float)
        fitted_steps = []
        for name, step in steps:
            obj = clone(step)
            if not hasattr(obj, "fit") or not hasattr(obj, "transform"):
                raise TypeError(f"Preprocessing step '{name}' must define fit and transform")
            Xt = obj.fit(Xt, y).transform(Xt)
            fitted_steps.append((name, obj))
        self.pipeline_ = fitted_steps
        return Xt

    def _transform_preprocessing(self, X):
        Xt = np.asarray(X, dtype=float)
        if self.pipeline_ is None:
            return Xt
        for _, step in self.pipeline_:
            Xt = step.transform(Xt)
        return Xt

    def _prepare_supervised(self, X, y=None):
        if y is not None:
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)
            if X.ndim != 2:
                raise ValueError("When y is provided, X must be a 2D feature matrix")
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must contain the same number of samples")
            return X, y, None

        series = _as_1d_series(X)
        features, target = lagged_matrix(series, lags=self.lags, horizon=self.horizon)
        return features, target, series

    def fit(self, X, y=None):
        features, target, history = self._prepare_supervised(X, y)
        self.history_ = history
        self.feature_transformer_ = LaggedTimeSeriesTransformer(lags=self.lags, horizon=self.horizon) if y is None else None
        Xt = self._fit_transform_preprocessing(features, target)
        self.model_ = clone(self.model)
        self.model_.fit(Xt, target)
        self.training_features_ = Xt
        self.training_target_ = target
        self.mode_ = "series" if y is None else "supervised"
        return self

    def _require_fitted(self):
        if self.model_ is None:
            raise RuntimeError("TimeSeriesPipeline has not been fit yet")

    def _features_from_input(self, X):
        if np.asarray(X).ndim == 2:
            return np.asarray(X, dtype=float)
        series = _as_1d_series(X)
        return lagged_matrix(series, lags=self.lags, horizon=self.horizon)[0]

    def predict(self, X=None):
        self._require_fitted()
        if X is None:
            if self.history_ is None:
                raise ValueError("X must be provided when the pipeline was fit in supervised mode")
            X = self.history_
        features = self._features_from_input(X)
        Xt = self._transform_preprocessing(features)
        return self.model_.predict(Xt)

    def forecast(self, steps=1, history=None):
        self._require_fitted()
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if history is None:
            if self.history_ is None:
                raise ValueError("history must be provided when the pipeline was fit in supervised mode")
            history = self.history_
        values = list(_as_1d_series(history))
        if len(values) < self.lags:
            raise ValueError("history must contain at least lags values")
        outputs = []
        for _ in range(int(steps)):
            window = np.asarray(values[-self.lags :], dtype=float).reshape(1, -1)
            Xt = self._transform_preprocessing(window)
            pred = float(np.asarray(self.model_.predict(Xt)).ravel()[0])
            outputs.append(pred)
            values.append(pred)
        return np.asarray(outputs, dtype=float)

    def score(self, X, y=None):
        self._require_fitted()
        if y is None:
            if np.asarray(X).ndim == 1:
                _, target = lagged_matrix(X, lags=self.lags, horizon=self.horizon)
                pred = self.predict(X)
                return -mean_squared_error(target, pred)
            raise ValueError("y must be provided when scoring on precomputed features")
        features = self._features_from_input(X)
        Xt = self._transform_preprocessing(features)
        if hasattr(self.model_, "score"):
            return self.model_.score(Xt, y)
        pred = self.model_.predict(Xt)
        return -mean_squared_error(y, pred)


def _grid(params):
    if not isinstance(params, dict) or not params:
        raise ValueError("param_grid must be a non-empty dict")
    keys = list(params)
    vals = []
    for key in keys:
        cur = list(params[key])
        if not cur:
            raise ValueError(f"Parameter grid for '{key}' must be non-empty")
        vals.append(cur)
    out = [{}]
    for key, cur in zip(keys, vals):
        nxt = []
        for base in out:
            for val in cur:
                item = dict(base)
                item[key] = val
                nxt.append(item)
        out = nxt
    return out


def _sample(params, n_iter, seed=None):
    if not isinstance(params, dict) or not params:
        raise ValueError("param_distributions must be a non-empty dict")
    rng = np.random.default_rng(seed)
    keys = list(params)
    out = []
    for _ in range(int(n_iter)):
        item = {}
        for key in keys:
            cur = list(params[key])
            if not cur:
                raise ValueError(f"Parameter distribution for '{key}' must be non-empty")
            item[key] = cur[int(rng.integers(0, len(cur)))]
        out.append(item)
    return out


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


class TimeSeriesExperiment(BaseEstimator):
    """Fit, tune, and score a forecasting model on ordered series splits."""

    def __init__(
        self,
        model,
        lags=12,
        horizon=1,
        preprocessing=None,
        search="grid",
        param_grid=None,
        param_distributions=None,
        n_iter=10,
        cv=None,
        scoring="neg_mean_squared_error",
        refit=True,
        random_state=None,
    ):
        self.model = model
        self.lags = int(lags)
        self.horizon = int(horizon)
        self.preprocessing = preprocessing
        self.search = search
        self.param_grid = param_grid
        self.param_distributions = param_distributions
        self.n_iter = int(n_iter)
        self.cv = cv
        self.scoring = scoring
        self.refit = bool(refit)
        self.random_state = random_state
        self.pipeline_ = None
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.history_ = None

    def _split(self, x):
        if self.cv is None:
            from ..data.cv import TimeSeriesSplit

            cv = TimeSeriesSplit(n_splits=5)
        elif isinstance(self.cv, int):
            from ..data.cv import TimeSeriesSplit

            cv = TimeSeriesSplit(n_splits=int(self.cv))
        else:
            cv = self.cv
        try:
            return list(cv.split(x))
        except TypeError:
            return list(cv.split(x, x))

    def _cands(self):
        if self.search is None:
            return [{}]
        if self.search == "grid":
            return _grid(self.param_grid)
        if self.search == "random":
            return _sample(self.param_distributions, self.n_iter, seed=self.random_state)
        raise ValueError("search must be one of: None, 'grid', 'random'")

    def _make_pipe(self, params=None):
        pipe = TimeSeriesPipeline(
            model=clone(self.model),
            lags=self.lags,
            horizon=self.horizon,
            preprocessing=self.preprocessing,
        )
        if params:
            pipe.set_params(**params)
        return pipe

    def fit(self, X, y=None):
        x = _as_1d_series(X)
        cand = self._cands()
        splits = self._split(x)
        rows = []
        best = None
        best_s = -np.inf

        for prm in cand:
            scores = []
            for tr, te in splits:
                p = self._make_pipe(prm)
                hist = x[tr]
                test = x[te]
                p.fit(hist)
                pred = p.forecast(steps=test.size, history=hist)
                scores.append(_score(test, pred, self.scoring))
            s = float(np.mean(scores))
            row = {"params": prm, "mean_test_score": s, "test_scores": np.asarray(scores, dtype=float)}
            rows.append(row)
            if s > best_s:
                best_s = s
                best = prm

        self.cv_results_ = rows
        self.best_params_ = best
        self.best_score_ = best_s
        if self.refit:
            self.best_estimator_ = self._make_pipe(best)
            self.best_estimator_.fit(x)
            self.pipeline_ = self.best_estimator_
        self.history_ = x
        return self

    def _need(self):
        if self.best_estimator_ is None:
            raise RuntimeError("TimeSeriesExperiment has not been fit yet")

    def forecast(self, steps=1, history=None):
        self._need()
        return self.best_estimator_.forecast(steps=steps, history=history)

    def predict(self, X=None):
        self._need()
        if X is None:
            if self.history_ is None:
                raise RuntimeError("No training history available")
            X = self.history_
        return self.best_estimator_.predict(X)

    def score(self, X, y=None):
        self._need()
        if y is None:
            hist = self.history_
            if hist is None:
                raise RuntimeError("No training history available")
            tgt = hist[self.lags :]
            pred = self.best_estimator_.predict(hist)
            return _score(tgt, pred, self.scoring)
        hist = _as_1d_series(X)
        tgt = np.asarray(y, dtype=float).ravel()
        pred = self.forecast(steps=tgt.size, history=hist)
        return _score(tgt, pred, self.scoring)

    def summary(self):
        self._need()
        return {
            "lags": self.lags,
            "horizon": self.horizon,
            "search": self.search,
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "model": self.model.__class__.__name__,
            "has_preprocessing": self.preprocessing is not None,
        }


__all__ = [
    "ARModel",
    "LaggedTimeSeriesTransformer",
    "TimeSeriesExperiment",
    "TimeSeriesPipeline",
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
