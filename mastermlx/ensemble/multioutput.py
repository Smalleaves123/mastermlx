from __future__ import annotations

import inspect
import numpy as np

from ..base import BaseEstimator
from ..utils import clone
from ..utils.validation import check_X, check_sample_weight, check_y


def _fit_with_sample_weight(estimator, X, y, sample_weight):
    parameters = inspect.signature(estimator.fit).parameters
    if "sample_weight" in parameters or any(
        parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        estimator.fit(X, y)
    return estimator


class MultiOutputClassifier(BaseEstimator):
    """Fit one classifier per target column (multi-label / multi-output)."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.estimators_ = []
        self.classes_ = []

    def fit(self, X, y=None, sample_weight=None):
        X = check_X(X)
        y = check_y(y, allow_2d=True)
        if y.ndim == 1:
            y = y[:, None]
        if y.shape[1] == 0:
            raise ValueError("y must contain at least one output")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        sample_weight = check_sample_weight(sample_weight, X.shape[0])
        self._set_n_features(X)

        self.estimators_ = []
        self.classes_ = []
        for j in range(y.shape[1]):
            yj = y[:, j]
            uniq = np.unique(yj)
            if uniq.size == 0:
                raise ValueError(f"Target column {j} has no unique values")
            est = clone(self.estimator)
            _fit_with_sample_weight(est, X, yj, sample_weight)
            self.estimators_.append(est)
            self.classes_.append(np.unique(yj) if hasattr(est, 'classes_') else uniq)
        return self

    def predict(self, X):
        X = self._check_X(X)
        if not self.estimators_:
            raise RuntimeError("Model has not been fit yet")
        preds = [est.predict(X) for est in self.estimators_]
        return np.column_stack([np.asarray(p, dtype=preds[0].dtype).ravel() for p in preds])

    def predict_proba(self, X):
        X = self._check_X(X)
        if not self.estimators_:
            raise RuntimeError("Model has not been fit yet")
        return [est.predict_proba(X) for est in self.estimators_ if hasattr(est, 'predict_proba')]

    def score(self, X, y, sample_weight=None):
        y = check_y(y, allow_2d=True)
        if y.ndim == 1:
            y = y[:, None]
        pred = self.predict(X)
        if y.shape != pred.shape:
            raise ValueError("y and predictions must have the same shape")
        weights = check_sample_weight(sample_weight, y.shape[0])
        return float(np.average(np.all(y == pred, axis=1), weights=weights))


class MultiOutputRegressor(BaseEstimator):
    """Fit one regressor per target column."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.estimators_ = []

    def fit(self, X, y=None, sample_weight=None):
        X = check_X(X)
        y = check_y(y, allow_2d=True, dtype=float, ensure_all_finite=True)
        if y.ndim == 1:
            y = y[:, None]
        if y.shape[1] == 0:
            raise ValueError("y must contain at least one output")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        sample_weight = check_sample_weight(sample_weight, X.shape[0])
        self._set_n_features(X)

        self.estimators_ = []
        for j in range(y.shape[1]):
            est = clone(self.estimator)
            _fit_with_sample_weight(est, X, y[:, j], sample_weight)
            self.estimators_.append(est)
        return self

    def predict(self, X):
        X = self._check_X(X)
        if not self.estimators_:
            raise RuntimeError("Model has not been fit yet")
        preds = [est.predict(X) for est in self.estimators_]
        return np.column_stack([np.asarray(p, dtype=float).ravel() for p in preds])

    def score(self, X, y, sample_weight=None):
        y = check_y(y, allow_2d=True, dtype=float, ensure_all_finite=True)
        pred = self.predict(X)
        if y.shape[0] != pred.shape[0]:
            raise ValueError("X and y must contain the same number of rows")
        weights = check_sample_weight(sample_weight, y.shape[0])
        if y.ndim == 1:
            mean = np.average(y, weights=weights)
            denom = np.sum(weights * (y - mean) ** 2)
            return float(1.0 - np.sum(weights * (y - pred.ravel()) ** 2) / denom) if denom else 0.0
        values = []
        for j in range(y.shape[1]):
            mean = np.average(y[:, j], weights=weights)
            denom = np.sum(weights * (y[:, j] - mean) ** 2)
            values.append(1.0 - np.sum(weights * (y[:, j] - pred[:, j]) ** 2) / denom if denom else 0.0)
        return float(np.mean(values))
