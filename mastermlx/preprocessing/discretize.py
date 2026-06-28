from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils.validation import check_2d_array


class KBinsDiscretizer(BaseTransformer):
    """Bin continuous features into discrete intervals."""

    def __init__(self, n_bins=5, strategy="quantile", encode="ordinal"):
        self.n_bins = int(n_bins)
        self.strategy = strategy
        self.encode = encode
        self.bin_edges_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        if self.strategy not in {"uniform", "quantile"}:
            raise ValueError("strategy must be one of: uniform, quantile")
        if self.encode not in {"ordinal", "onehot"}:
            raise ValueError("encode must be one of: ordinal, onehot")

        edges = []
        for j in range(X.shape[1]):
            col = X[:, j]
            if self.strategy == "uniform":
                cur = np.linspace(np.min(col), np.max(col), self.n_bins + 1)
            else:
                cur = np.quantile(col, np.linspace(0.0, 1.0, self.n_bins + 1))
            cur = np.asarray(cur, dtype=float)
            for idx in range(1, cur.shape[0]):
                if cur[idx] <= cur[idx - 1]:
                    cur[idx] = cur[idx - 1] + 1e-12
            edges.append(cur)
        self.bin_edges_ = edges
        return self

    def transform(self, X):
        X = check_2d_array(X).astype(float)
        if self.bin_edges_ is None:
            raise RuntimeError("KBinsDiscretizer has not been fit yet")
        if X.shape[1] != len(self.bin_edges_):
            raise ValueError("X has a different number of features than the fitted data")

        codes = np.zeros_like(X, dtype=int)
        for j, edges in enumerate(self.bin_edges_):
            bins = np.digitize(X[:, j], edges[1:-1], right=False)
            bins = np.clip(bins, 0, self.n_bins - 1)
            codes[:, j] = bins

        if self.encode == "ordinal":
            return codes.astype(float)

        parts = []
        eye = np.eye(self.n_bins, dtype=float)
        for j in range(codes.shape[1]):
            parts.append(eye[codes[:, j]])
        return np.hstack(parts)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
