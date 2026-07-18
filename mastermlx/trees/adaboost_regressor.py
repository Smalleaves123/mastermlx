from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import as_2d, check_1d_array, check_2d_array, r2_score
from .decision_tree import DecisionTreeRegressor


class AdaBoostRegressor(BaseEstimator):
    """Boosted regressor with shallow trees and adaptive sample weights."""

    def __init__(self, n_estimators=50, learning_rate=1.0, max_depth=2,
                 min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = np.empty(0, dtype=float)
        self.init_ = 0.0

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y).astype(float)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

        n_samples = X.shape[0]
        sample_weight = np.full(n_samples, 1.0 / n_samples, dtype=float)
        self.init_ = float(np.average(y, weights=sample_weight))
        current = np.full(n_samples, self.init_, dtype=float)
        self.estimators_ = []
        estimator_weights: list[float] = []

        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X, y - current)
            update = tree.predict(X)
            current += self.learning_rate * update
            residual = np.abs(y - current)
            scale = np.mean(residual) + 1e-12
            sample_weight = np.exp(np.clip(residual / scale, 0.0, 20.0))
            sample_weight /= np.sum(sample_weight)

            self.estimators_.append(tree)
            estimator_weights.append(self.learning_rate)
            if np.max(np.abs(update)) <= 1e-12:
                break

        self.estimator_weights_ = np.asarray(estimator_weights, dtype=float)
        return self

    def predict(self, X):
        if not self.estimators_:
            raise RuntimeError("Model has not been fit yet")

        X = as_2d(X).astype(float)
        pred = np.full(X.shape[0], self.init_, dtype=float)
        for alpha, tree in zip(self.estimator_weights_, self.estimators_):
            pred += alpha * tree.predict(X)
        return float(pred[0]) if pred.shape[0] == 1 else pred

    def score(self, X, y):
        return r2_score(y, self.predict(X))
