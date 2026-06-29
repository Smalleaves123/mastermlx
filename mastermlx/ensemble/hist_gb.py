from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import accuracy, as_2d, check_1d_array, check_2d_array, r2_score


def _bin_data(X, n_bins=256):
    """Quantile-bin each feature column. Returns binned X (0..n_bins-1) and bin edges."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    n_bins = min(n_bins, n)
    binned = np.zeros((n, d), dtype=np.int32)
    edges = []
    for j in range(d):
        col = X[:, j]
        uniq = np.unique(col)
        if uniq.size <= n_bins:
            # Use unique values directly
            mapping = {v: i for i, v in enumerate(uniq)}
            binned[:, j] = np.array([mapping[v] for v in col], dtype=np.int32)
            edges.append(uniq)
        else:
            percentiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.percentile(col, percentiles)
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            binned[:, j] = np.digitize(col, bin_edges[:-1]) - 1
            binned[:, j] = np.clip(binned[:, j], 0, n_bins - 1)
            edges.append(bin_edges)
    return binned, edges


class _HistNode:
    def __init__(self):
        self.left = self.right = None
        self.feat = -1
        self.bin_idx = -1
        self.value = 0.0


class _HistTree:
    def __init__(self, max_depth=6, min_samples_leaf=20, l2_reg=0.0, n_bins=256):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_reg = l2_reg
        self.n_bins = n_bins
        self.root = None

    def fit(self, X, g, h):
        """X: binned int32 (n, d). g, h: gradients and hessians (n,)."""
        n, d = X.shape
        self.n_bins = min(self.n_bins, n)
        self.root = self._grow(X, g, h, np.arange(n), 0)

    def _grow(self, X, g, h, indices, depth):
        n = len(indices)
        if n == 0:
            leaf = _HistNode()
            leaf.value = 0.0
            return leaf
        g_sum = np.sum(g[indices])
        h_sum = np.sum(h[indices])
        h_sum = max(h_sum, 1e-12)

        if depth >= self.max_depth or n < self.min_samples_leaf * 2:
            leaf = _HistNode()
            leaf.value = -g_sum / (h_sum + self.l2_reg)
            return leaf

        best_gain = -1e308
        best_feat = -1
        best_bin = -1

        for feat in range(X.shape[1]):
            col = X[indices, feat].astype(np.int32)
            max_bin = np.max(col) + 1
            if max_bin <= 1:
                continue
            # Build histogram of gradient sums per bin
            hist_g = np.zeros(max_bin, dtype=float)
            hist_h = np.zeros(max_bin, dtype=float)
            for k, idx in enumerate(indices):
                b = col[k]
                hist_g[b] += g[idx]
                hist_h[b] += h[idx]

            left_g, left_h = 0.0, 0.0
            for b in range(max_bin - 1):
                left_g += hist_g[b]
                left_h += hist_h[b]
                right_g = g_sum - left_g
                right_h = h_sum - left_h
                if left_h < self.min_samples_leaf or right_h < self.min_samples_leaf:
                    continue
                gain = (left_g * left_g) / (left_h + self.l2_reg) + \
                       (right_g * right_g) / (right_h + self.l2_reg) - \
                       (g_sum * g_sum) / (h_sum + self.l2_reg)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_bin = b

        if best_feat == -1:
            leaf = _HistNode()
            leaf.value = -g_sum / (h_sum + self.l2_reg)
            return leaf

        col = X[indices, best_feat].astype(np.int32)
        left_mask = col <= best_bin
        left_idx = indices[left_mask]
        right_idx = indices[~left_mask]

        if len(left_idx) == 0 or len(right_idx) == 0:
            leaf = _HistNode()
            leaf.value = -g_sum / (h_sum + self.l2_reg)
            return leaf

        node = _HistNode()
        node.feat = best_feat
        node.bin_idx = best_bin
        node.left = self._grow(X, g, h, left_idx, depth + 1)
        node.right = self._grow(X, g, h, right_idx, depth + 1)
        return node

    def predict(self, X):
        """X: binned int32 (n, d). Returns predictions (n,)."""
        n = X.shape[0]
        pred = np.zeros(n, dtype=float)
        for i in range(n):
            node = self.root
            while node.left is not None and node.right is not None:
                if X[i, node.feat] <= node.bin_idx:
                    node = node.left
                else:
                    node = node.right
            pred[i] = node.value
        return pred


class _HistGBBase(BaseEstimator):
    def __init__(self, loss, n_estimators=100, learning_rate=0.1, max_depth=6,
                 min_samples_leaf=20, l2_reg=0.0, max_bins=256, random_state=None):
        self.loss = loss
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_reg = float(l2_reg)
        self.max_bins = int(max_bins)
        self.random_state = random_state
        self.init_ = 0.0
        self.trees_ = []
        self._edges = None

    def _grad_hess(self, y_true, raw_pred):
        raise NotImplementedError

    def fit(self, X, y):
        X = check_2d_array(X)
        y = check_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows")

        X_binned, self._edges = _bin_data(X, self.max_bins)
        raw_pred = np.full(y.shape[0], self.init_, dtype=float)
        self.trees_ = []

        for _ in range(self.n_estimators):
            g, h = self._grad_hess(y, raw_pred)
            tree = _HistTree(self.max_depth, self.min_samples_leaf, self.l2_reg, self.max_bins)
            tree.fit(X_binned, g, h)
            update = tree.predict(X_binned)
            raw_pred += self.learning_rate * update
            self.trees_.append(tree)
        return self

    def predict_raw(self, X):
        X = check_2d_array(X)
        X_binned = np.zeros((X.shape[0], X.shape[1]), dtype=np.int32)
        for j, edges in enumerate(self._edges):
            if edges is not None:
                X_binned[:, j] = np.digitize(X[:, j], edges[:-1]) - 1
                X_binned[:, j] = np.clip(X_binned[:, j], 0, self.max_bins - 1)
        raw = np.full(X.shape[0], self.init_, dtype=float)
        for tree in self.trees_:
            raw += self.learning_rate * tree.predict(X_binned)
        return raw


class HistGradientBoostingClassifier(_HistGBBase):
    """Histogram-based gradient boosting for classification (binary log-loss)."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6,
                 min_samples_leaf=20, l2_reg=0.0, max_bins=256, random_state=None):
        super().__init__(loss="log_loss", n_estimators=n_estimators,
                         learning_rate=learning_rate, max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf, l2_reg=l2_reg,
                         max_bins=max_bins, random_state=random_state)
        self.classes_ = None

    def _grad_hess(self, y_true, raw_pred):
        y_true = np.asarray(y_true, dtype=float)
        p = 1.0 / (1.0 + np.exp(-raw_pred))
        g = p - y_true
        h = p * (1.0 - p)
        h = np.maximum(h, 1e-12)
        return g, h

    def fit(self, X, y):
        X = check_2d_array(X)
        y = check_1d_array(y)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("HistGradientBoostingClassifier only supports binary classification")
        y_bin = (y == self.classes_[1]).astype(float)
        return super().fit(X, y_bin)

    def predict_proba(self, X):
        raw = self.predict_raw(X)
        p1 = 1.0 / (1.0 + np.exp(-raw))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def score(self, X, y):
        return accuracy(y, self.predict(X))


class HistGradientBoostingRegressor(_HistGBBase):
    """Histogram-based gradient boosting for regression (squared error)."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6,
                 min_samples_leaf=20, l2_reg=0.0, max_bins=256, random_state=None):
        super().__init__(loss="squared_error", n_estimators=n_estimators,
                         learning_rate=learning_rate, max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf, l2_reg=l2_reg,
                         max_bins=max_bins, random_state=random_state)

    def _grad_hess(self, y_true, raw_pred):
        y_true = np.asarray(y_true, dtype=float)
        g = raw_pred - y_true
        h = np.ones_like(y_true)
        return g, h

    def fit(self, X, y):
        X = check_2d_array(X)
        y = check_1d_array(y).astype(float)
        self.init_ = float(np.mean(y))
        return super().fit(X, y)

    def predict(self, X):
        return self.predict_raw(X)

    def score(self, X, y):
        return r2_score(y, self.predict(X))
