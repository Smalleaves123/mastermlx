from __future__ import annotations

import math
import numpy as np

from ..base import BaseEstimator
from ..utils import accuracy, as_2d, check_1d_array, check_2d_array, clone, r2_score
from ._base import _clone_list, _majority, _mean_pred


class _BagBase(BaseEstimator):
    def __init__(self, estimator, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, random_state=None):
        self.estimator = estimator
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.random_state = random_state
        self.estimators_ = []
        self.rows_ = []
        self.cols_ = []

    def _n_take(self, value, n, name):
        if isinstance(value, float):
            if not 0 < value <= 1.0:
                raise ValueError(f"{name} as float must be in (0, 1]")
            return max(1, int(round(value * n)))
        value = int(value)
        if value < 1 or value > n:
            raise ValueError(f"{name} must be between 1 and n_samples/n_features")
        return value

    def _take_cols(self, m, rng):
        k = self._n_take(self.max_features, m, "max_features")
        if self.bootstrap_features:
            cols = rng.integers(0, m, size=k)
        else:
            cols = rng.choice(m, size=k, replace=False)
        return np.asarray(cols, dtype=int)

    def _take_rows(self, n, rng):
        k = self._n_take(self.max_samples, n, "max_samples")
        if self.bootstrap:
            rows = rng.integers(0, n, size=k)
        else:
            rows = rng.choice(n, size=k, replace=False)
        return np.asarray(rows, dtype=int)


class BaggingClassifier(_BagBase):
    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        self.rows_ = []
        self.cols_ = []
        for _ in range(self.n_estimators):
            rows = self._take_rows(X.shape[0], rng)
            cols = self._take_cols(X.shape[1], rng)
            est = clone(self.estimator)
            est.fit(X[rows][:, cols], y[rows])
            self.estimators_.append(est)
            self.rows_.append(rows)
            self.cols_.append(cols)
        return self

    def predict(self, X):
        if not self.estimators_:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        preds = np.asarray([est.predict(X[:, cols]) for est, cols in zip(self.estimators_, self.cols_)])
        out = _majority(preds, None)
        return out

    def score(self, X, y):
        return accuracy(y, self.predict(X))


class BaggingRegressor(_BagBase):
    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y).astype(float)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        self.rows_ = []
        self.cols_ = []
        for _ in range(self.n_estimators):
            rows = self._take_rows(X.shape[0], rng)
            cols = self._take_cols(X.shape[1], rng)
            est = clone(self.estimator)
            est.fit(X[rows][:, cols], y[rows])
            self.estimators_.append(est)
            self.rows_.append(rows)
            self.cols_.append(cols)
        return self

    def predict(self, X):
        if not self.estimators_:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        preds = np.asarray([est.predict(X[:, cols]) for est, cols in zip(self.estimators_, self.cols_)], dtype=float)
        out = np.mean(preds, axis=0)
        return float(out[0]) if out.shape[0] == 1 else out

    def score(self, X, y):
        return r2_score(y, self.predict(X))
