from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import as_2d, check_2d_array


class AgglomerativeClustering(BaseEstimator):
    """Hierarchical agglomerative clustering."""

    def __init__(self, n_clusters=2, linkage="ward"):
        self.n_clusters = int(n_clusters)
        self.linkage = linkage
        self.labels_ = None
        self.children_ = None
        self.cluster_centers_ = None
        self.n_leaves_ = 0

    def _pair_score(self, X, clusters, i, j):
        a = clusters[i]
        b = clusters[j]

        if self.linkage == "single":
            diff = X[a][:, None, :] - X[b][None, :, :]
            return float(np.min(np.sqrt(np.sum(diff * diff, axis=2))))
        if self.linkage == "complete":
            diff = X[a][:, None, :] - X[b][None, :, :]
            return float(np.max(np.sqrt(np.sum(diff * diff, axis=2))))
        if self.linkage == "average":
            diff = X[a][:, None, :] - X[b][None, :, :]
            return float(np.mean(np.sqrt(np.sum(diff * diff, axis=2))))
        if self.linkage == "ward":
            ca = np.mean(X[a], axis=0)
            cb = np.mean(X[b], axis=0)
            na = len(a)
            nb = len(b)
            delta = ca - cb
            return float((na * nb) / (na + nb) * np.sum(delta * delta))
        raise ValueError("linkage must be one of: single, complete, average, ward")

    def _merge(self, clusters, i, j):
        merged = clusters[i] + clusters[j]
        new_clusters = []
        for idx, cluster in enumerate(clusters):
            if idx not in {i, j}:
                new_clusters.append(cluster)
        new_clusters.append(merged)
        return new_clusters, merged

    def fit(self, X, y=None):
        X = check_2d_array(X)
        n_samples = X.shape[0]
        if self.n_clusters < 1 or self.n_clusters > n_samples:
            raise ValueError("n_clusters must be between 1 and the number of samples")
        if self.linkage not in {"single", "complete", "average", "ward"}:
            raise ValueError("linkage must be one of: single, complete, average, ward")

        clusters = [[i] for i in range(n_samples)]
        children = []

        while len(clusters) > self.n_clusters:
            best_i = None
            best_j = None
            best_score = np.inf
            for i in range(len(clusters) - 1):
                for j in range(i + 1, len(clusters)):
                    score = self._pair_score(X, clusters, i, j)
                    if score < best_score:
                        best_score = score
                        best_i = i
                        best_j = j
            clusters, merged = self._merge(clusters, best_i, best_j)
            children.append((best_i, best_j))

        labels = np.empty(n_samples, dtype=int)
        centers = []
        for cluster_id, cluster in enumerate(clusters):
            labels[cluster] = cluster_id
            centers.append(np.mean(X[cluster], axis=0))

        self.labels_ = labels
        self.children_ = np.asarray(children, dtype=int) if children else np.empty((0, 2), dtype=int)
        self.cluster_centers_ = np.asarray(centers)
        self.n_leaves_ = n_samples
        self.n_clusters_ = len(clusters)
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


AggloClust = AgglomerativeClustering
