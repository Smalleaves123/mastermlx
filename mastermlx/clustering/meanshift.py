from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import as_2d, check_2d_array


def _squared_distances(X, Y):
    diff = X[:, None, :] - Y[None, :, :]
    return np.sum(diff * diff, axis=2)


class MeanShift(BaseEstimator):
    """Mean shift clustering with a flat kernel."""

    def __init__(self, bandwidth=1.0, max_iter=300, tol=1e-3):
        self.bandwidth = float(bandwidth)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.cluster_centers_ = None
        self.labels_ = None
        self.n_clusters_ = 0
        self.n_iter_ = 0

    def _shift_point(self, X, point):
        dist_sq = np.sum((X - point) ** 2, axis=1)
        mask = dist_sq <= self.bandwidth ** 2
        if not np.any(mask):
            return point
        return np.mean(X[mask], axis=0)

    def _merge_centers(self, centers):
        unique = []
        for center in centers:
            if not unique:
                unique.append(center)
                continue
            d = np.sqrt(np.sum((np.asarray(unique) - center) ** 2, axis=1))
            if np.min(d) > self.bandwidth * 0.5:
                unique.append(center)
        return np.asarray(unique)

    def fit(self, X, y=None):
        X = check_2d_array(X)
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")

        shifted = []
        max_iter_seen = 0
        for seed in X:
            center = seed.copy()
            for it in range(1, self.max_iter + 1):
                new_center = self._shift_point(X, center)
                shift = np.linalg.norm(new_center - center)
                center = new_center
                if shift < self.tol:
                    break
            shifted.append(center)
            max_iter_seen = max(max_iter_seen, it)

        centers = self._merge_centers(np.asarray(shifted))
        if centers.size == 0:
            centers = np.mean(X, axis=0, keepdims=True)

        dist_sq = _squared_distances(X, centers)
        labels = np.argmin(dist_sq, axis=1)

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.n_clusters_ = centers.shape[0]
        self.n_iter_ = max_iter_seen
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        if X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError("X has a different number of features than the fitted data")
        dist_sq = _squared_distances(X, self.cluster_centers_)
        labels = np.argmin(dist_sq, axis=1)
        return labels[0] if labels.shape[0] == 1 else labels
