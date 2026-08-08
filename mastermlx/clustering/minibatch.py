from __future__ import annotations

import numpy as np

from ..accel.ml_kernels import kmeans_assign, kmeans_update
from ..base import BaseEstimator
from ..utils import as_2d, check_2d_array
from ..utils.random import resolve_rng


class MiniBatchKMeans(BaseEstimator):
    """KMeans with mini-batch SGD updates for large datasets."""

    def __init__(self, n_clusters=8, batch_size=100, max_iter=100, n_init=3,
                 tol=1e-4, random_state=None):
        self.n_clusters = int(n_clusters)
        self.batch_size = int(batch_size)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.tol = float(tol)
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.counts_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        n, d = X.shape
        if self.n_clusters < 1 or self.n_clusters > n:
            raise ValueError("n_clusters must be between 1 and the number of samples")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if self.n_init < 1:
            raise ValueError("n_init must be at least 1")
        k = self.n_clusters
        batch_size = min(self.batch_size, n)
        rng = resolve_rng(self.random_state)
        best_inertia = np.inf

        for init_run in range(self.n_init):
            idx = rng.choice(n, size=k, replace=False)
            centers = X[idx].copy()
            counts = np.ones(k, dtype=float)

            for _ in range(self.max_iter):
                batch_idx = rng.choice(n, size=batch_size, replace=False)
                Xb = X[batch_idx]
                labels_b, _ = kmeans_assign(Xb, centers)
                sums, batch_counts = kmeans_update(Xb, labels_b, k)

                for j in range(k):
                    if batch_counts[j] == 0:
                        continue
                    lr = 1.0 / max(counts[j], 1.0)
                    centers[j] = (1.0 - lr) * centers[j] + lr * sums[j] / batch_counts[j]
                    counts[j] += batch_counts[j]

            labels, sq_all = self._assign(X, centers)
            inertia = float(np.sum(sq_all))
            if inertia < best_inertia:
                best_inertia = inertia
                self.cluster_centers_ = centers
                self.labels_ = labels
                self.counts_ = counts
        self.inertia_ = best_inertia
        return self

    def _assign(self, X, centers):
        return kmeans_assign(X, centers)

    def predict(self, X):
        X = as_2d(X).astype(float)
        if self.cluster_centers_ is None:
            raise RuntimeError("not fitted")
        labels, _ = self._assign(X, self.cluster_centers_)
        return labels

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_
