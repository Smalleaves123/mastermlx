from __future__ import annotations

import numpy as np
from typing import cast

from ..base import BaseEstimator
from ..utils import accuracy, as_2d, check_1d_array, check_2d_array, r2_score


class _ExtraNode:
    def __init__(self, *, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None


class _ExtraTreeBase:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root_ = None

    def _pick_features(self, m, rng):
        if self.max_features is None:
            k = m
        elif self.max_features == "sqrt":
            k = max(1, int(np.sqrt(m)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(m)))
        elif isinstance(self.max_features, float):
            if not 0 < self.max_features <= 1:
                raise ValueError("max_features as float must be in (0, 1]")
            k = max(1, int(round(self.max_features * m)))
        else:
            k = int(self.max_features)
        if k < 1 or k > m:
            raise ValueError("max_features must select between 1 and n_features")
        return np.asarray(rng.choice(m, size=k, replace=False), dtype=int)

    def _leaf_value(self, y):
        raise NotImplementedError

    def _score_split(self, yl, yr, *args):
        raise NotImplementedError

    def _grow(self, X, y, depth, rng):
        n, m = X.shape
        stop = n < self.min_samples_split or np.unique(y).shape[0] == 1 or (self.max_depth is not None and depth >= self.max_depth)
        if stop:
            return _ExtraNode(value=self._leaf_value(y))

        feats = self._pick_features(m, rng)
        best_feat = None
        best_thr = None
        best_score = np.inf

        for feat in feats:
            x = X[:, feat]
            lo = np.min(x)
            hi = np.max(x)
            if lo == hi:
                continue
            for _ in range(16):
                thr = rng.uniform(lo, hi)
                left = x <= thr
                right = ~left
                nl = left.sum()
                nr = right.sum()
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                score = self._score_split(y[left], y[right], nl, nr, n)
                if score < best_score:
                    best_score = score
                    best_feat = feat
                    best_thr = thr

        if best_feat is None:
            return _ExtraNode(value=self._leaf_value(y))

        threshold = cast(float, best_thr)
        left = X[:, best_feat] <= threshold
        right = ~left
        if left.sum() == 0 or right.sum() == 0:
            return _ExtraNode(value=self._leaf_value(y))

        node = _ExtraNode(feature=best_feat, threshold=threshold)
        node.left = self._grow(X[left], y[left], depth + 1, rng)
        node.right = self._grow(X[right], y[right], depth + 1, rng)
        return node

    def _predict_one(self, x, node):
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value


class ExtraTreeClassifier(_ExtraTreeBase):
    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.random_state)
        self.root_ = self._grow(X, y, 0, rng)
        return self

    def _leaf_value(self, y):
        vals, cnt = np.unique(y, return_counts=True)
        return vals[np.argmax(cnt)]

    def _score_split(self, yl, yr, *args):
        nl, nr, n = args
        def gini(y):
            _, cnt = np.unique(y, return_counts=True)
            p = cnt / cnt.sum()
            return 1.0 - np.sum(p ** 2)

        return (nl / n) * gini(yl) + (nr / n) * gini(yr)

    def predict(self, X):
        if self.root_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        pred = np.array([self._predict_one(x, self.root_) for x in X])
        return pred[0] if pred.shape[0] == 1 else pred

    def score(self, X, y):
        return accuracy(y, self.predict(X))


class ExtraTreeRegressor(_ExtraTreeBase):
    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y).astype(float)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        rng = np.random.default_rng(self.random_state)
        self.root_ = self._grow(X, y, 0, rng)
        return self

    def _leaf_value(self, y):
        return float(np.mean(y))

    def _score_split(self, yl, yr, *args):
        nl, nr, n = args
        return (nl / n) * np.var(yl) + (nr / n) * np.var(yr)

    def predict(self, X):
        if self.root_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        pred = np.array([self._predict_one(x, self.root_) for x in X], dtype=float)
        return float(pred[0]) if pred.shape[0] == 1 else pred

    def score(self, X, y):
        return r2_score(y, self.predict(X))


class ExtraTreesClassifier(BaseEstimator):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", random_state=None):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        for _ in range(self.n_estimators):
            tree = ExtraTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=rng.integers(0, 2**32 - 1),
            )
            tree.fit(X, y)
            self.trees_.append(tree)
        return self

    def predict(self, X):
        if not self.trees_:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X).astype(float)
        preds = np.asarray([tree.predict(X) for tree in self.trees_])
        out: list[object] = []
        for col in preds.T:
            vals, cnt = np.unique(col, return_counts=True)
            out.append(vals[np.argmax(cnt)])
        result = np.asarray(out)
        return result[0] if result.shape[0] == 1 else result

    def score(self, X, y):
        return accuracy(y, self.predict(X))


class ExtraTreesRegressor(BaseEstimator):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", random_state=None):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y).astype(float)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        for _ in range(self.n_estimators):
            tree = ExtraTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=rng.integers(0, 2**32 - 1),
            )
            tree.fit(X, y)
            self.trees_.append(tree)
        return self

    def predict(self, X):
        if not self.trees_:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X).astype(float)
        preds = np.asarray([tree.predict(X) for tree in self.trees_], dtype=float)
        out = np.mean(preds, axis=0)
        return float(out[0]) if out.shape[0] == 1 else out

    def score(self, X, y):
        return r2_score(y, self.predict(X))
