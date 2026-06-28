from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import as_2d, check_2d_array
from .kmeans import KMeans


def _pairwise_sq_dists(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.sum(diff * diff, axis=2)


class SpectralClustering(BaseEstimator):
    """Spectral clustering with an RBF or nearest-neighbor affinity."""

    def __init__(
        self,
        n_clusters=2,
        affinity="rbf",
        gamma=None,
        n_neighbors=10,
        assign_labels="kmeans",
        random_state=None,
    ):
        self.n_clusters = int(n_clusters)
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = int(n_neighbors)
        self.assign_labels = assign_labels
        self.random_state = random_state
        self.affinity_matrix_ = None
        self.embedding_ = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.n_clusters_ = None

    def _resolve_gamma(self, n_features):
        if self.gamma is None or self.gamma == "scale":
            return 1.0 / max(n_features, 1)
        return float(self.gamma)

    def _build_affinity(self, X):
        if self.affinity == "rbf":
            gamma = self._resolve_gamma(X.shape[1])
            dist_sq = _pairwise_sq_dists(X)
            W = np.exp(-gamma * dist_sq)
            np.fill_diagonal(W, 0.0)
            return W
        if self.affinity == "nearest_neighbors":
            if self.n_neighbors < 1:
                raise ValueError("n_neighbors must be at least 1")
            dist_sq = _pairwise_sq_dists(X)
            W = np.zeros_like(dist_sq)
            for i in range(X.shape[0]):
                idx = np.argsort(dist_sq[i])[1 : self.n_neighbors + 1]
                W[i, idx] = 1.0
            W = np.maximum(W, W.T)
            return W
        raise ValueError("affinity must be 'rbf' or 'nearest_neighbors'")

    def fit(self, X, y=None):
        X = check_2d_array(X)
        if self.n_clusters < 1 or self.n_clusters > X.shape[0]:
            raise ValueError("n_clusters must be between 1 and the number of samples")
        if self.assign_labels != "kmeans":
            raise ValueError("assign_labels must be 'kmeans'")

        W = self._build_affinity(X)
        D = np.sum(W, axis=1)
        D_inv_sqrt = np.where(D > 0, 1.0 / np.sqrt(D), 0.0)
        L = np.eye(X.shape[0]) - (D_inv_sqrt[:, None] * W * D_inv_sqrt[None, :])

        eigvals, eigvecs = np.linalg.eigh(L)
        embedding = eigvecs[:, : self.n_clusters].copy()
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = np.divide(embedding, norms, out=np.zeros_like(embedding), where=norms > 0)

        km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        labels = km.fit_predict(embedding)

        centers = []
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            if np.any(mask):
                centers.append(np.mean(X[mask], axis=0))
            else:
                centers.append(np.mean(X, axis=0))

        self.affinity_matrix_ = W
        self.embedding_ = embedding
        self.labels_ = labels
        self.cluster_centers_ = np.asarray(centers)
        self.n_clusters_ = self.n_clusters
        self.eigenvalues_ = eigvals[: self.n_clusters]
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        if X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError("X has a different number of features than the fitted data")
        diff = X[:, None, :] - self.cluster_centers_[None, :, :]
        labels = np.argmin(np.sum(diff * diff, axis=2), axis=1)
        return labels[0] if labels.shape[0] == 1 else labels


SpecClust = SpectralClustering
