from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils.validation import check_1d_array, check_2d_array, check_same_rows


class QuantileRegressor(BaseEstimator):
    """Linear quantile regression with pinball loss via iteratively reweighted least squares."""

    def __init__(self, quantile=0.5, alpha=0.0, fit_intercept=True, max_iter=100, tol=1e-6):
        self.quantile = float(quantile)
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y).astype(float)
        X, y = check_same_rows(X, y)
        if not 0 < self.quantile < 1:
            raise ValueError("quantile must be in (0, 1)")
        n, d = X.shape
        tau = self.quantile

        if self.fit_intercept:
            Xb = np.column_stack([np.ones(n), X])
        else:
            Xb = X

        # Start from OLS
        beta = np.linalg.lstsq(Xb, y, rcond=None)[0]
        prev = beta.copy()

        for _ in range(self.max_iter):
            resid = y - Xb @ beta
            # Pinball subgradient weights: τ if resid>0, (τ-1) otherwise
            # For IRLS: weight = τ/|resid| if resid>0, (1-τ)/|resid| if resid<0
            abs_resid = np.abs(resid)
            w = np.where(resid > 0, tau, 1.0 - tau) / np.maximum(abs_resid, 1e-8)
            # Normalize weights
            w = w / np.sum(w) * n

            W = np.diag(w)
            XTW = Xb.T @ W
            A = XTW @ Xb
            if self.alpha > 0:
                A += self.alpha * np.eye(Xb.shape[1])
            beta = np.linalg.solve(A, XTW @ y)

            if np.max(np.abs(beta - prev)) < self.tol:
                break
            prev = beta.copy()

        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta
        return self

    def predict(self, X):
        X = check_2d_array(X).astype(float)
        if self.coef_ is None:
            raise RuntimeError("Model has not been fit yet")
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        y = check_1d_array(y).astype(float)
        pred = self.predict(X)
        resid = y - pred
        tau = self.quantile
        return -float(np.mean(np.where(resid >= 0, tau * resid, (tau - 1.0) * resid)))
