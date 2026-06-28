from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils.metrics import r2_score
from ..utils.validation import check_2d_array


class LinearRegression(BaseEstimator):
    """Ordinary least squares linear regression."""

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        if self.fit_intercept:
            X_aug = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_aug = X

        params, *_ = np.linalg.lstsq(X_aug, y, rcond=None)

        if self.fit_intercept:
            self.intercept_ = float(params[0])
            self.coef_ = params[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = params

        return self

    def predict(self, X):
        X = check_2d_array(X)
        if self.coef_ is None:
            raise RuntimeError("Model has not been fit yet")
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        return r2_score(y, self.predict(X))

