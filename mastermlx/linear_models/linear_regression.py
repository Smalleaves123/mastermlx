from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils.metrics import r2_score
from ..utils.validation import check_X_y


class LinearRegression(BaseEstimator):
    """Ordinary least squares linear regression."""

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y=None):
        X, y = check_X_y(X, y)
        self._set_n_features(X)

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
        self._check_fitted(["coef_", "intercept_"])
        X = self._check_X(X)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        return r2_score(y, self.predict(X))
