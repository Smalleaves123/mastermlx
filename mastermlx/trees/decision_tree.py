from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import accuracy, as_2d, check_1d_array, check_2d_array, r2_score


class _Node:
    def __init__(self, *, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier(BaseEstimator):
    """A simple CART-style decision tree for classification."""

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root_ = None
        self.classes_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        if self.min_samples_split < 2 * self.min_samples_leaf:
            raise ValueError("min_samples_split must be at least twice min_samples_leaf")

        self.classes_ = np.unique(y)
        self.root_ = self._grow(X, y, depth=0)
        return self

    def _gini(self, y):
        vals, cnt = np.unique(y, return_counts=True)
        p = cnt / cnt.sum()
        return 1.0 - np.sum(p ** 2)

    def _best_split(self, X, y):
        from ..accel.backends import best_split_classifier
        result = best_split_classifier(X.astype(float), y.astype(np.int64), self.min_samples_leaf)
        if result[0] is not None:
            return result

        # Fallback: pure NumPy implementation
        n, m = X.shape
        best_feat = None
        best_thr = None
        best_score = np.inf

        base = self._gini(y)
        if base == 0.0:
            return None, None, None

        for j in range(m):
            x = X[:, j]
            uniq = np.unique(x)
            if uniq.shape[0] < 2:
                continue
            thr = (uniq[:-1] + uniq[1:]) / 2.0
            for t in thr:
                left = x <= t
                right = ~left
                nl = left.sum()
                nr = right.sum()
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                score = (nl / n) * self._gini(y[left]) + (nr / n) * self._gini(y[right])
                if score < best_score:
                    best_score = score
                    best_feat = j
                    best_thr = t

        if best_feat is None:
            return None, None, None
        return best_feat, best_thr, best_score

    def _leaf_value(self, y):
        vals, cnt = np.unique(y, return_counts=True)
        return vals[np.argmax(cnt)]

    def _grow(self, X, y, depth):
        n = X.shape[0]
        stop = (
            n < self.min_samples_split
            or np.unique(y).shape[0] == 1
            or (self.max_depth is not None and depth >= self.max_depth)
        )
        if stop:
            return _Node(value=self._leaf_value(y))

        feat, thr, _ = self._best_split(X, y)
        if feat is None:
            return _Node(value=self._leaf_value(y))

        left = X[:, feat] <= thr
        right = ~left
        if left.sum() == 0 or right.sum() == 0:
            return _Node(value=self._leaf_value(y))

        node = _Node(feature=feat, threshold=thr)
        node.left = self._grow(X[left], y[left], depth + 1)
        node.right = self._grow(X[right], y[right], depth + 1)
        return node

    def _predict_one(self, x, node):
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        if self.root_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        if X.shape[1] is None:
            raise ValueError("Invalid input shape")
        pred = np.array([self._predict_one(x, self.root_) for x in X])
        return pred[0] if pred.shape[0] == 1 else pred

    def score(self, X, y):
        return accuracy(y, self.predict(X))


class DecisionTreeRegressor(BaseEstimator):
    """A simple CART-style decision tree for regression."""

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        if self.min_samples_split < 2 * self.min_samples_leaf:
            raise ValueError("min_samples_split must be at least twice min_samples_leaf")

        self.root_ = self._grow(X, y.astype(float), depth=0)
        return self

    def _variance(self, y):
        if y.size == 0:
            return 0.0
        return float(np.var(y))

    def _best_split(self, X, y):
        from ..accel.backends import best_split_regressor
        result = best_split_regressor(X.astype(float), y.astype(float), self.min_samples_leaf)
        if result[0] is not None:
            return result

        # Fallback: pure NumPy
        n, m = X.shape
        best_feat = None
        best_thr = None
        best_score = np.inf

        base = self._variance(y)
        if base == 0.0:
            return None, None, None

        for j in range(m):
            x = X[:, j]
            uniq = np.unique(x)
            if uniq.shape[0] < 2:
                continue
            thr = (uniq[:-1] + uniq[1:]) / 2.0
            for t in thr:
                left = x <= t
                right = ~left
                nl = left.sum()
                nr = right.sum()
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                score = (nl / n) * self._variance(y[left]) + (nr / n) * self._variance(y[right])
                if score < best_score:
                    best_score = score
                    best_feat = j
                    best_thr = t

        if best_feat is None:
            return None, None, None
        return best_feat, best_thr, best_score

    def _leaf_value(self, y):
        return float(np.mean(y))

    def _grow(self, X, y, depth):
        n = X.shape[0]
        stop = (
            n < self.min_samples_split
            or np.unique(y).shape[0] == 1
            or (self.max_depth is not None and depth >= self.max_depth)
        )
        if stop:
            return _Node(value=self._leaf_value(y))

        feat, thr, _ = self._best_split(X, y)
        if feat is None:
            return _Node(value=self._leaf_value(y))

        left = X[:, feat] <= thr
        right = ~left
        if left.sum() == 0 or right.sum() == 0:
            return _Node(value=self._leaf_value(y))

        node = _Node(feature=feat, threshold=thr)
        node.left = self._grow(X[left], y[left], depth + 1)
        node.right = self._grow(X[right], y[right], depth + 1)
        return node

    def _predict_one(self, x, node):
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        if self.root_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        pred = np.array([self._predict_one(x, self.root_) for x in X], dtype=float)
        return float(pred[0]) if pred.shape[0] == 1 else pred

    def score(self, X, y):
        return r2_score(y, self.predict(X))
