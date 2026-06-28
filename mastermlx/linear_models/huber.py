from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils.validation import check_2d_array, check_1d_array, check_same_rows


class HuberRegressor(BaseEstimator):
    """Linear regression with Huber loss, robust to outliers.

    Solved via iteratively reweighted least squares (IRLS).
    """

    def __init__(self, epsilon=1.35, max_iter=50, tol=1e-5, alpha=0.0, fit_intercept=True):
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.coef_ = None
        self.intercept_ = 0.0
        self.weights_ = None
        self.outliers_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y).astype(float)
        X, y = check_same_rows(X, y)
        n, d = X.shape

        # Start with OLS
        if self.fit_intercept:
            X_aug = np.column_stack([np.ones(n), X])
        else:
            X_aug = X

        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        prev_beta = beta.copy()
        scale = np.std(y - X_aug @ beta)
        scale = max(scale, 1e-12)

        for _ in range(self.max_iter):
            resid = y - X_aug @ beta
            # Huber weights: clip at epsilon * scale
            w = np.where(np.abs(resid) <= self.epsilon * scale, 1.0,
                         self.epsilon * scale / np.abs(resid))

            # Weighted least squares with optional L2 penalty
            if self.fit_intercept:
                W = np.diag(w)
                XTW = X_aug.T @ W
                A = XTW @ X_aug
                if self.alpha > 0:
                    aug = self.alpha * np.eye(d + 1)
                    aug[0, 0] = 0  # Don't penalize intercept
                    A = A + aug
                beta = np.linalg.solve(A, XTW @ y)
            else:
                W = np.diag(w)
                XTW = X_aug.T @ W
                A = XTW @ X_aug
                if self.alpha > 0:
                    A = A + self.alpha * np.eye(d)
                beta = np.linalg.solve(A, XTW @ y)

            # Update scale
            resid = y - X_aug @ beta
            scale = np.std(resid[w > 0.5]) if np.any(w > 0.5) else scale
            scale = max(scale, 1e-12)

            if np.max(np.abs(beta - prev_beta)) < self.tol:
                break
            prev_beta = beta.copy()

        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta

        self.weights_ = w
        self.outliers_ = np.flatnonzero(w < 0.5)
        return self

    def predict(self, X):
        X = check_2d_array(X).astype(float)
        if self.coef_ is None:
            raise RuntimeError("Model has not been fit yet")
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        from ..utils.metrics import r2_score
        return r2_score(y, self.predict(X))
