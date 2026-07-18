from __future__ import annotations


from ..base import BaseTransformer
from ._core import center_dist, eig_embed, pairwise_dist


class MDS(BaseTransformer):
    """Classical multidimensional scaling."""

    def __init__(self, n_components=2):
        self.n_components = int(n_components)
        self.embedding_ = None
        self.eigenvalues_ = None
        self.distances_ = None

    def fit(self, X, y=None):
        D = pairwise_dist(X)
        B = center_dist(D)
        emb, vals = eig_embed(B, self.n_components, high=True)
        self.embedding_ = emb
        self.eigenvalues_ = vals
        self.distances_ = D
        return self

    def transform(self, X):
        return self.fit_transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).embedding_


ClassicalMDS = MDS
