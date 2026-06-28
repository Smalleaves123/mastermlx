from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import as_2d, check_1d_array, check_2d_array, check_same_rows, r2_score


class LinearSVR(BaseEstimator):
    """Linear support vector regressor trained with subgradient descent."""

    def __init__(self, C=1.0, epsilon=0.1, lr=0.01, max_iter=2000, tol=1e-6, fit_intercept=True, random_state=None):
        self.C = float(C)
        self.epsilon = float(epsilon)
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self.loss_ = []
        self.mean_ = None
        self.scale_ = None

    def _add_bias(self, X):
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y).astype(float)
        X, y = check_same_rows(X, y)
        if self.C <= 0:
            raise ValueError("C must be positive")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")

        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0

        Xn = (X - self.mean_) / self.scale_
        Xb = self._add_bias(Xn)
        rng = np.random.default_rng(self.random_state)
        w = rng.normal(scale=0.01, size=Xb.shape[1])
        self.loss_ = []
        prev = None

        for _ in range(self.max_iter):
            pred = Xb @ w
            err = pred - y
            abs_err = np.abs(err)

            loss = 0.5 * np.sum(w[1:] ** 2 if self.fit_intercept else w ** 2)
            loss += self.C * np.sum(np.maximum(0.0, abs_err - self.epsilon))
            self.loss_.append(float(loss))

            grad = np.zeros_like(w)
            if self.fit_intercept:
                grad[1:] = w[1:]
            else:
                grad = w.copy()

            active = abs_err > self.epsilon
            if np.any(active):
                signs = np.sign(err[active])
                grad += self.C * (Xb[active].T @ signs)

            w -= self.lr * grad / Xb.shape[0]

            if prev is not None and abs(prev - loss) < self.tol:
                break
            prev = loss

        if self.fit_intercept:
            coef = w[1:] / self.scale_
            self.coef_ = coef
            self.intercept_ = float(w[0] - self.mean_ @ coef)
        else:
            self.intercept_ = 0.0
            self.coef_ = w / self.scale_
        return self

    def predict(self, X):
        X = as_2d(X)
        if self.coef_ is None:
            raise RuntimeError("Model has not been fit yet")
        pred = X @ self.coef_ + self.intercept_
        return float(pred[0]) if pred.shape[0] == 1 else pred

    def score(self, X, y):
        return r2_score(y, self.predict(X))
