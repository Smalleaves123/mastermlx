from __future__ import annotations

import numpy as np

from ..accel import pairwise_distances
from ..base import BaseEstimator
from ..utils import check_2d_array


def _pairwise_distances(X):
    return pairwise_distances(X, X)


class DBSCAN(BaseEstimator):
    """Density-based clustering with a simple NumPy implementation."""

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.labels_ = None
        self.core_sample_indices_ = None
        self.components_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X)
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.min_samples < 1:
            raise ValueError("min_samples must be at least 1")

        distances = _pairwise_distances(X)
        neighbors = [np.flatnonzero(distances[i] <= self.eps) for i in range(X.shape[0])]
        core_mask = np.array([nn.size >= self.min_samples for nn in neighbors], dtype=bool)
        core_samples = np.flatnonzero(core_mask)

        labels = np.full(X.shape[0], -1, dtype=int)
        cluster_id = 0
        visited = np.zeros(X.shape[0], dtype=bool)

        for point in range(X.shape[0]):
            if visited[point] or not core_mask[point]:
                continue

            stack = [point]
            visited[point] = True
            labels[point] = cluster_id

            while stack:
                current = stack.pop()
                for neighbor in neighbors[current]:
                    if labels[neighbor] == -1:
                        labels[neighbor] = cluster_id
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        if core_mask[neighbor]:
                            stack.append(neighbor)
            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = core_samples
        self.components_ = X[core_samples]
        self.n_clusters_ = cluster_id
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_

    def predict(self, X):
        if self.labels_ is None:
            raise RuntimeError("Model has not been fit yet")
        raise NotImplementedError("DBSCAN does not support predicting labels for new samples")
