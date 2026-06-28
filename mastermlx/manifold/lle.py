from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ._core import eig_embed, kgraph, pairwise_dist


class LLE(BaseTransformer):
    """Locally linear embedding."""

    def __init__(self, n_components=2, n_neighbors=5, reg=1e-3):
        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.reg = float(reg)
        self.embedding_ = None
        self.weights_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        n = X.shape[0]
        if self.n_components < 1 or self.n_components >= n:
            raise ValueError("n_components must be between 1 and n_samples - 1")

        D = pairwise_dist(X)
        G = kgraph(D, self.n_neighbors)
        nbr = np.argsort(G, axis=1)[:, 1 : self.n_neighbors + 1]

        W = np.zeros((n, n), dtype=float)
        eye = np.eye(self.n_neighbors)
        for i in range(n):
            idx = nbr[i]
            Z = X[idx] - X[i]
            C = Z @ Z.T
            C += self.reg * np.trace(C) * eye
            w = np.linalg.solve(C, np.ones(self.n_neighbors))
            w /= np.sum(w)
            W[i, idx] = w

        M = np.eye(n) - W
        M = M.T @ M
        emb, _ = eig_embed(M, self.n_components + 1, high=False, scale=False)
        self.embedding_ = emb[:, 1 : self.n_components + 1]
        self.weights_ = W
        return self

    def transform(self, X):
        return self.fit_transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).embedding_


LocallyLinearEmbedding = LLE
