from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import clone
from ..utils.validation import check_2d_array


class MultiOutputClassifier(BaseEstimator):
    """Fit one classifier per target column (multi-label / multi-output)."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.estimators_ = []
        self.classes_ = []

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2:
            raise ValueError("y must be 1D or 2D")
        if y.shape[0] == 0:
            raise ValueError("y must be non-empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        self.estimators_ = []
        self.classes_ = []
        for j in range(y.shape[1]):
            yj = y[:, j]
            uniq = np.unique(yj)
            if uniq.size == 0:
                raise ValueError(f"Target column {j} has no unique values")
            est = clone(self.estimator)
            est.fit(X, yj)
            self.estimators_.append(est)
            self.classes_.append(np.unique(yj) if hasattr(est, 'classes_') else uniq)
        return self

    def predict(self, X):
        X = check_2d_array(X)
        if not self.estimators_:
            raise RuntimeError("Model has not been fit yet")
        preds = [est.predict(X) for est in self.estimators_]
        return np.column_stack([np.asarray(p, dtype=preds[0].dtype).ravel() for p in preds])

    def predict_proba(self, X):
        X = check_2d_array(X)
        if not self.estimators_:
            raise RuntimeError("Model has not been fit yet")
        return [est.predict_proba(X) for est in self.estimators_ if hasattr(est, 'predict_proba')]


class MultiOutputRegressor(BaseEstimator):
    """Fit one regressor per target column."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.estimators_ = []

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2:
            raise ValueError("y must be 1D or 2D")
        if y.shape[0] == 0:
            raise ValueError("y must be non-empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        self.estimators_ = []
        for j in range(y.shape[1]):
            est = clone(self.estimator)
            est.fit(X, y[:, j])
            self.estimators_.append(est)
        return self

    def predict(self, X):
        X = check_2d_array(X)
        if not self.estimators_:
            raise RuntimeError("Model has not been fit yet")
        preds = [est.predict(X) for est in self.estimators_]
        return np.column_stack([np.asarray(p, dtype=float).ravel() for p in preds])

    def score(self, X, y):
        from ..utils.metrics import r2_score
        y = np.asarray(y, dtype=float)
        pred = self.predict(X)
        if y.ndim == 1:
            return r2_score(y, pred.ravel())
        return float(np.mean([r2_score(y[:, j], pred[:, j]) for j in range(y.shape[1])]))
