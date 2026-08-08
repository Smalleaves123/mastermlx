from __future__ import annotations

import numpy as np

from ..accel.ml_kernels import meanshift_update
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
        unique: list[np.ndarray] = []
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
        if not np.isfinite(self.bandwidth) or self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive and finite")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol must be non-negative and finite")

        shifted = np.asarray(X, dtype=float).copy()
        active = np.ones(X.shape[0], dtype=bool)
        iterations = np.zeros(X.shape[0], dtype=int)
        for it in range(1, self.max_iter + 1):
            active_indices = np.flatnonzero(active)
            if active_indices.size == 0:
                break
            previous = shifted[active_indices].copy()
            updated = meanshift_update(X, previous, self.bandwidth)
            shifted[active_indices] = updated
            shifts = np.linalg.norm(updated - previous, axis=1)
            done = (shifts < self.tol) | (it == self.max_iter)
            iterations[active_indices[done]] = it
            active[active_indices[done]] = False
        max_iter_seen = int(np.max(iterations)) if iterations.size else 0

        centers = self._merge_centers(shifted)
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
        return labels
