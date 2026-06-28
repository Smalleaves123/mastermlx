from __future__ import annotations

import math
import numpy as np

from ..base import BaseEstimator
from ..utils import accuracy, as_2d, check_1d_array, check_2d_array, r2_score
from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestClassifier(BaseEstimator):
    """Random forest classifier built from decision trees."""

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees_ = []
        self.cols_ = []
        self.classes_ = None

    def _pick_cols(self, n_feat, rng):
        if self.max_features is None:
            k = n_feat
        elif self.max_features == "sqrt":
            k = max(1, int(math.sqrt(n_feat)))
        elif self.max_features == "log2":
            k = max(1, int(math.log2(n_feat)))
        elif isinstance(self.max_features, float):
            if not 0 < self.max_features <= 1:
                raise ValueError("max_features as float must be in (0, 1]")
            k = max(1, int(round(self.max_features * n_feat)))
        elif isinstance(self.max_features, int):
            k = self.max_features
        else:
            raise ValueError("Unsupported max_features value")

        if k < 1 or k > n_feat:
            raise ValueError("max_features must select between 1 and n_features")
        return np.sort(rng.choice(n_feat, size=k, replace=False))

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")

        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        self.cols_ = []
        self.classes_ = np.unique(y)

        n, m = X.shape
        for _ in range(self.n_estimators):
            cols = self._pick_cols(m, rng)
            if self.bootstrap:
                idx = rng.integers(0, n, size=n)
                X_sub = X[idx][:, cols]
                y_sub = y[idx]
            else:
                X_sub = X[:, cols]
                y_sub = y

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_sub, y_sub)
            self.trees_.append(tree)
            self.cols_.append(cols)

        return self

    def predict(self, X):
        if not self.trees_:
            raise RuntimeError("Model has not been fit yet")

        X = as_2d(X)
        pred = []
        for tree, cols in zip(self.trees_, self.cols_):
            pred.append(tree.predict(X[:, cols]))
        pred = np.asarray(pred)

        out = []
        for j in range(pred.shape[1]):
            vals, cnt = np.unique(pred[:, j], return_counts=True)
            out.append(vals[np.argmax(cnt)])

        out = np.asarray(out)
        return out[0] if out.shape[0] == 1 else out

    def score(self, X, y):
        return accuracy(y, self.predict(X))


class RandomForestRegressor(BaseEstimator):
    """Random forest regressor built from decision trees."""

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees_ = []
        self.cols_ = []

    def _pick_cols(self, n_feat, rng):
        if self.max_features is None:
            k = n_feat
        elif self.max_features == "sqrt":
            k = max(1, int(math.sqrt(n_feat)))
        elif self.max_features == "log2":
            k = max(1, int(math.log2(n_feat)))
        elif isinstance(self.max_features, float):
            if not 0 < self.max_features <= 1:
                raise ValueError("max_features as float must be in (0, 1]")
            k = max(1, int(round(self.max_features * n_feat)))
        elif isinstance(self.max_features, int):
            k = self.max_features
        else:
            raise ValueError("Unsupported max_features value")

        if k < 1 or k > n_feat:
            raise ValueError("max_features must select between 1 and n_features")
        return np.sort(rng.choice(n_feat, size=k, replace=False))

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")

        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        self.cols_ = []

        n, m = X.shape
        for _ in range(self.n_estimators):
            cols = self._pick_cols(m, rng)
            if self.bootstrap:
                idx = rng.integers(0, n, size=n)
                X_sub = X[idx][:, cols]
                y_sub = y[idx]
            else:
                X_sub = X[:, cols]
                y_sub = y

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_sub, y_sub)
            self.trees_.append(tree)
            self.cols_.append(cols)

        return self

    def predict(self, X):
        if not self.trees_:
            raise RuntimeError("Model has not been fit yet")

        X = as_2d(X)
        pred = []
        for tree, cols in zip(self.trees_, self.cols_):
            pred.append(tree.predict(X[:, cols]))
        pred = np.asarray(pred, dtype=float)
        out = np.mean(pred, axis=0)
        return float(out[0]) if out.shape[0] == 1 else out

    def score(self, X, y):
        return r2_score(y, self.predict(X))


RFClf = RandomForestClassifier
RFReg = RandomForestRegressor
