from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import as_2d, check_1d_array, check_2d_array, r2_score
from .decision_tree import DecisionTreeRegressor


class GradientBoostingRegressor(BaseEstimator):
    """Gradient boosting regressor with squared-error loss."""

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
    ):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.init_ = 0.0
        self.estimators_ = []

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y).astype(float)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

        self.init_ = float(np.mean(y))
        current_pred = np.full(y.shape[0], self.init_, dtype=float)
        self.estimators_ = []

        for _ in range(self.n_estimators):
            residual = y - current_pred
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X, residual)
            update = tree.predict(X)
            current_pred += self.learning_rate * update
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        if not self.estimators_:
            raise RuntimeError("Model has not been fit yet")

        X = as_2d(X).astype(float)
        pred = np.full(X.shape[0], self.init_, dtype=float)
        for tree in self.estimators_:
            pred += self.learning_rate * tree.predict(X)
        return float(pred[0]) if pred.shape[0] == 1 else pred

    def score(self, X, y):
        return r2_score(y, self.predict(X))


GBR = GradientBoostingRegressor
