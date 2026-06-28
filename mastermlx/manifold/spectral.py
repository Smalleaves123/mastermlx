from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ._core import eig_embed, kgraph, pairwise_dist


class SpectralEmbedding(BaseTransformer):
    """Spectral embedding from a k-nearest-neighbor graph."""

    def __init__(self, n_components=2, n_neighbors=5, gamma=None):
        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.gamma = gamma
        self.embedding_ = None
        self.affinity_ = None

    def _scale(self, D):
        if self.gamma is None:
            return 1.0
        if self.gamma == "scale":
            m = np.mean(D[D > 0]) if np.any(D > 0) else 1.0
            return 1.0 / max(m ** 2, 1e-12)
        return float(self.gamma)

    def fit(self, X, y=None):
        D = pairwise_dist(X)
        G = kgraph(D, self.n_neighbors)
        A = np.zeros_like(G)
        s = self._scale(D)
        mask = np.isfinite(G) & (G > 0)
        A[mask] = np.exp(-s * G[mask] ** 2)
        A = np.maximum(A, A.T)
        np.fill_diagonal(A, 0.0)

        deg = np.sum(A, axis=1)
        if np.any(deg <= 0):
            raise ValueError("graph must be connected")
        d = 1.0 / np.sqrt(deg)
        L = np.eye(A.shape[0]) - (d[:, None] * A) * d[None, :]
        emb, _ = eig_embed(L, self.n_components + 1, high=False, scale=False)
        self.embedding_ = emb[:, 1 : self.n_components + 1]
        self.affinity_ = A
        return self

    def transform(self, X):
        return self.fit_transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).embedding_
