from __future__ import annotations

import math
import numpy as np


def average_path_length(n):
    n = int(n)
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1.0) + 0.5772156649) - 2.0 * (n - 1.0) / n


class _IsolationNode:
    def __init__(self, *, feature=None, threshold=None, left=None, right=None, size=0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.size = int(size)

    @property
    def is_leaf(self):
        return self.left is None and self.right is None


class IsolationTree:
    """Random isolation tree used inside IsolationForest."""

    def __init__(self, max_depth, random_state=None):
        self.max_depth = int(max_depth)
        self.random_state = random_state
        self.root_ = None
        self.features_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.root_ = self._grow(X, depth=0, rng=rng)
        return self

    def _grow(self, X, depth, rng):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples <= 1:
            return _IsolationNode(size=n_samples)

        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        span = maxs - mins
        valid = np.flatnonzero(span > 0.0)
        if valid.size == 0:
            return _IsolationNode(size=n_samples)

        feat = int(valid[rng.integers(0, valid.size)])
        thr = float(rng.uniform(mins[feat], maxs[feat]))
        left_mask = X[:, feat] < thr
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return _IsolationNode(size=n_samples)

        return _IsolationNode(
            feature=feat,
            threshold=thr,
            left=self._grow(X[left_mask], depth + 1, rng),
            right=self._grow(X[right_mask], depth + 1, rng),
            size=n_samples,
        )

    def path_length(self, x):
        if self.root_ is None:
            raise RuntimeError("Tree has not been fit yet")
        node = self.root_
        depth = 0
        while not node.is_leaf:
            if x[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
            depth += 1
        return depth + average_path_length(node.size)

    def path_lengths(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self.path_length(row) for row in X], dtype=float)
