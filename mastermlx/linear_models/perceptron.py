from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils.validation import check_2d_array, check_1d_array, check_same_rows


class Perceptron(BaseEstimator):
    """Classic Rosenblatt perceptron for binary classification."""

    def __init__(self, max_iter=1000, eta0=1.0, random_state=None):
        self.max_iter = int(max_iter)
        self.eta0 = float(eta0)
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = 0.0
        self.classes_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y)
        X, y = check_same_rows(X, y)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("Perceptron only supports binary classification")
        y_bin = np.where(y == self.classes_[1], 1.0, -1.0)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        if self.coef_ is None:
            self.coef_ = rng.normal(scale=0.01, size=d)
        self.intercept_ = 0.0

        for _ in range(self.max_iter):
            errors = 0
            order = rng.permutation(n)
            for i in order:
                margin = y_bin[i] * (X[i] @ self.coef_ + self.intercept_)
                if margin <= 0:
                    self.coef_ += self.eta0 * y_bin[i] * X[i]
                    self.intercept_ += self.eta0 * y_bin[i]
                    errors += 1
            if errors == 0:
                break
        return self

    def predict(self, X):
        X = check_2d_array(X).astype(float)
        coef = self.coef_
        classes = self.classes_
        if coef is None or classes is None:
            raise RuntimeError("Model has not been fit yet")
        scores = X @ coef + self.intercept_
        return np.where(scores >= 0, classes[1], classes[0])

    def decision_function(self, X):
        X = check_2d_array(X).astype(float)
        return X @ self.coef_ + self.intercept_
