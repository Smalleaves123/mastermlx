from __future__ import annotations

import numpy as np
from typing import Any, cast

from ..base import BaseEstimator
from ..utils.estimator import clone
from ..utils.metrics import mean_squared_error
from .ts_core import LaggedTimeSeriesTransformer, _as_1d_series, _normalize_steps, _score, lagged_matrix


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
        self.scoring = "neg_mean_squared_error"

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
        model = cast(Any, self.model_)
        if X is None:
            if self.history_ is None:
                raise ValueError("X must be provided when the pipeline was fit in supervised mode")
            X = self.history_
        features = self._features_from_input(X)
        Xt = self._transform_preprocessing(features)
        return model.predict(Xt)

    def forecast(self, steps=1, history=None):
        self._require_fitted()
        model = cast(Any, self.model_)
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
            pred = float(np.asarray(model.predict(Xt)).ravel()[0])
            outputs.append(pred)
            values.append(pred)
        return np.asarray(outputs, dtype=float)

    def score(self, X, y=None):
        self._require_fitted()
        model = cast(Any, self.model_)
        if y is None:
            if self.training_features_ is None or self.training_target_ is None:
                raise RuntimeError("No training history available")
            if hasattr(model, "score"):
                return float(model.score(self.training_features_, self.training_target_))
            pred = model.predict(self.training_features_)
            return _score(self.training_target_, pred, self.scoring)
        features = self._features_from_input(X)
        Xt = self._transform_preprocessing(features)
        if hasattr(model, "score"):
            return model.score(Xt, y)
        pred = model.predict(Xt)
        return -mean_squared_error(y, pred)




__all__ = ["ARModel", "TimeSeriesPipeline"]
