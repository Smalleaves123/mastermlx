from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils.validation import check_X, check_sample_weight, check_y, to_dense


class LinearRegression(BaseEstimator):
    """Ordinary least squares linear regression."""

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X, y=None, sample_weight=None):
        X = check_X(X, dtype=float, ensure_all_finite=True)
        y = check_y(y, allow_2d=True, dtype=float, ensure_all_finite=True)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        X = to_dense(X)
        sample_weight = check_sample_weight(sample_weight, X.shape[0])
        self._set_n_features(X)

        sqrt_weight = np.sqrt(sample_weight)
        if self.fit_intercept:
            X_aug = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_aug = X

        params, *_ = np.linalg.lstsq(
            X_aug * sqrt_weight[:, None],
            y * sqrt_weight[:, None] if y.ndim == 2 else y * sqrt_weight,
            rcond=None,
        )

        if self.fit_intercept:
            if y.ndim == 1:
                self.intercept_ = float(params[0])
                self.coef_ = params[1:]
            else:
                self.intercept_ = params[0]
                self.coef_ = params[1:].T
        else:
            self.intercept_ = 0.0 if y.ndim == 1 else np.zeros(y.shape[1])
            self.coef_ = params if y.ndim == 1 else params.T
        self.n_outputs_ = 1 if y.ndim == 1 else y.shape[1]

        return self

    def predict(self, X):
        self._check_fitted(["coef_", "intercept_"])
        X = to_dense(self._check_X(X, dtype=float, ensure_all_finite=True))
        if np.ndim(self.coef_) == 1:
            return X @ self.coef_ + self.intercept_
        return X @ self.coef_.T + self.intercept_

    def score(self, X, y, sample_weight=None):
        y = check_y(y, allow_2d=True, dtype=float, ensure_all_finite=True)
        pred = np.asarray(self.predict(X))
        if y.shape != pred.shape:
            raise ValueError("y and predictions must have the same shape")
        weights = check_sample_weight(sample_weight, y.shape[0])
        values = []
        targets = y.T if y.ndim == 2 else (y,)
        predictions = pred.T if pred.ndim == 2 else (pred,)
        for target, prediction in zip(targets, predictions):
            mean = np.average(target, weights=weights)
            ss_res = np.sum(weights * (target - prediction) ** 2)
            ss_tot = np.sum(weights * (target - mean) ** 2)
            values.append(1.0 - ss_res / ss_tot if ss_tot else 0.0)
        return float(np.mean(values))
