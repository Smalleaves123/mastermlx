from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import as_2d, check_2d_array


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
        k = min(self.n_clusters, n)
        rng = np.random.default_rng(self.random_state)
        best_inertia = np.inf

        for init_run in range(self.n_init):
            idx = rng.choice(n, size=k, replace=False)
            centers = X[idx].copy()
            counts = np.ones(k, dtype=float)

            for _ in range(self.max_iter):
                batch_idx = rng.choice(n, size=self.batch_size, replace=False)
                Xb = X[batch_idx]
                sq = np.sum(Xb**2, axis=1)[:, None] + np.sum(centers**2, axis=1)[None, :] \
                     - 2.0 * (Xb @ centers.T)
                labels_b = np.argmin(sq, axis=1)

                for j in range(k):
                    mask = labels_b == j
                    if np.sum(mask) == 0:
                        continue
                    lr = 1.0 / max(counts[j], 1.0)
                    centers[j] = (1.0 - lr) * centers[j] + lr * np.mean(Xb[mask], axis=0)
                    counts[j] += np.sum(mask)

            labels, sq_all = self._assign(X, centers)
            inertia = float(np.mean(sq_all[np.arange(n), labels]))
            if inertia < best_inertia:
                best_inertia = inertia
                self.cluster_centers_ = centers
                self.labels_ = labels
                self.counts_ = counts
        self.inertia_ = best_inertia
        return self

    def _assign(self, X, centers):
        sq = np.sum(X**2, axis=1)[:, None] + np.sum(centers**2, axis=1)[None, :] \
             - 2.0 * (X @ centers.T)
        return np.argmin(sq, axis=1), sq

    def predict(self, X):
        X = as_2d(X).astype(float)
        if self.cluster_centers_ is None:
            raise RuntimeError("not fitted")
        labels, _ = self._assign(X, self.cluster_centers_)
        return labels[0] if labels.shape[0] == 1 else labels

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_
