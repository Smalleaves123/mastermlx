from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils.metrics import r2_score
from ..utils.validation import check_1d_array, check_2d_array, check_same_rows


def _center_data(X, y, fit_intercept):
    if fit_intercept:
        X_mean = np.mean(X, axis=0)
        y_mean = float(np.mean(y))
        return X - X_mean, y - y_mean, X_mean, y_mean
    return X.copy(), y.copy(), np.zeros(X.shape[1], dtype=float), 0.0


def _soft_threshold(value, threshold):
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


class RidgeRegression(BaseEstimator):
    """Ridge regression solved with a closed-form linear system."""

    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y)
        X, y = check_same_rows(X, y)
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")

        Xc, yc, X_mean, y_mean = _center_data(X, y, self.fit_intercept)
        gram = Xc.T @ Xc
        penalty = self.alpha * np.eye(X.shape[1], dtype=float)
        self.coef_ = np.linalg.solve(gram + penalty, Xc.T @ yc)
        self.intercept_ = y_mean - X_mean @ self.coef_ if self.fit_intercept else 0.0
        return self

    def predict(self, X):
        X = check_2d_array(X)
        if self.coef_ is None:
            raise RuntimeError("Model has not been fit yet")
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        return r2_score(y, self.predict(X))


class LassoRegression(BaseEstimator):
    """Lasso regression solved with coordinate descent."""

    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y)
        X, y = check_same_rows(X, y)
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")

        Xc, yc, X_mean, y_mean = _center_data(X, y, self.fit_intercept)
        n_samples, n_features = X.shape
        w = np.zeros(n_features, dtype=float)
        col_norm_sq = np.sum(Xc * Xc, axis=0)

        for it in range(1, self.max_iter + 1):
            w_old = w.copy()
            for j in range(n_features):
                if col_norm_sq[j] == 0.0:
                    w[j] = 0.0
                    continue
                residual = yc - (Xc @ w) + Xc[:, j] * w[j]
                rho = Xc[:, j] @ residual
                w[j] = _soft_threshold(rho / col_norm_sq[j], self.alpha / col_norm_sq[j])
            if np.max(np.abs(w - w_old)) < self.tol:
                self.n_iter_ = it
                break
        else:
            self.n_iter_ = self.max_iter

        self.coef_ = w
        self.intercept_ = y_mean - X_mean @ self.coef_ if self.fit_intercept else 0.0
        return self

    def predict(self, X):
        X = check_2d_array(X)
        if self.coef_ is None:
            raise RuntimeError("Model has not been fit yet")
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        return r2_score(y, self.predict(X))


class ElasticNetRegression(BaseEstimator):
    """Elastic Net regression solved with coordinate descent."""

    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.fit_intercept = fit_intercept
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y)
        X, y = check_same_rows(X, y)
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if not 0.0 <= self.l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be in [0, 1]")

        Xc, yc, X_mean, y_mean = _center_data(X, y, self.fit_intercept)
        n_features = X.shape[1]
        w = np.zeros(n_features, dtype=float)
        col_norm_sq = np.sum(Xc * Xc, axis=0)
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1.0 - self.l1_ratio)

        for it in range(1, self.max_iter + 1):
            w_old = w.copy()
            for j in range(n_features):
                residual = yc - (Xc @ w) + Xc[:, j] * w[j]
                rho = Xc[:, j] @ residual
                denom = col_norm_sq[j] + l2
                if denom == 0.0:
                    w[j] = 0.0
                    continue
                w[j] = _soft_threshold(rho / denom, l1 / denom)
            if np.max(np.abs(w - w_old)) < self.tol:
                self.n_iter_ = it
                break
        else:
            self.n_iter_ = self.max_iter

        self.coef_ = w
        self.intercept_ = y_mean - X_mean @ self.coef_ if self.fit_intercept else 0.0
        return self

    def predict(self, X):
        X = check_2d_array(X)
        if self.coef_ is None:
            raise RuntimeError("Model has not been fit yet")
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        return r2_score(y, self.predict(X))
