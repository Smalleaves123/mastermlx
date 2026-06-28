from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ._core import all_pairs_shortest, center_dist, eig_embed, kgraph, pairwise_dist


class Isomap(BaseTransformer):
    """Isomap with dense all-pairs geodesic distances."""

    def __init__(self, n_components=2, n_neighbors=5):
        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.embedding_ = None
        self.geodesic_ = None
        self.distances_ = None

    def fit(self, X, y=None):
        D = pairwise_dist(X)
        G = kgraph(D, self.n_neighbors)
        G = all_pairs_shortest(G)
        B = center_dist(G)
        emb, _ = eig_embed(B, self.n_components, high=True)
        self.embedding_ = emb
        self.geodesic_ = G
        self.distances_ = D
        return self

    def transform(self, X):
        return self.fit_transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).embedding_
