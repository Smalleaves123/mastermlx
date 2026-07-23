from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..accel import pairwise_squared_euclidean
from ..base import BaseEstimator
from ..utils import as_2d, check_2d_array


def _squared_euclidean_distances(X, centers):
    return pairwise_squared_euclidean(X, centers)


class KMeans(BaseEstimator):
    """K-means clustering with k-means++ initialization."""

    def __init__(self, n_clusters=8, init="kmeans++", n_init=10, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _init_centers(self, X, rng):
        n_samples = X.shape[0]
        if self.n_clusters < 1 or self.n_clusters > n_samples:
            raise ValueError("n_clusters must be between 1 and the number of samples")
        if self.init not in {"kmeans++", "random"}:
            raise ValueError("init must be 'kmeans++' or 'random'")

        if self.init == "random":
            indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
            return X[indices].copy()

        centers = np.empty((self.n_clusters, X.shape[1]), dtype=float)
        first = int(rng.integers(0, n_samples))
        centers[0] = X[first]
        closest_dist_sq = _squared_euclidean_distances(X, centers[:1]).ravel()

        for i in range(1, self.n_clusters):
            probs = closest_dist_sq / np.sum(closest_dist_sq)
            idx = int(rng.choice(n_samples, p=probs))
            centers[i] = X[idx]
            dist_sq = _squared_euclidean_distances(X, centers[i : i + 1]).ravel()
            closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)

        return centers

    def _assign_labels(self, X, centers):
        dist_sq = _squared_euclidean_distances(X, centers)
        return np.argmin(dist_sq, axis=1), dist_sq

    def _update_centers(self, X, labels, old_centers, rng):
        centers = np.empty_like(old_centers)
        for j in range(self.n_clusters):
            mask = labels == j
            if np.any(mask):
                centers[j] = np.mean(X[mask], axis=0)
            else:
                centers[j] = X[int(rng.integers(0, X.shape[0]))]
        return centers

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "KMeans":
        X = check_2d_array(X)

        best_inertia = np.inf
        best_centers = None
        best_labels = None
        best_iter = 0

        n_runs = max(1, self.n_init)
        for run in range(n_runs):
            run_rng = np.random.default_rng(None if self.random_state is None else self.random_state + run)
            centers = self._init_centers(X, run_rng)
            prev_centers = centers.copy()
            prev_inertia = None

            for it in range(1, self.max_iter + 1):
                labels, dist_sq = self._assign_labels(X, centers)
                inertia = float(np.sum(dist_sq[np.arange(X.shape[0]), labels]))
                centers = self._update_centers(X, labels, centers, run_rng)
                shift = np.linalg.norm(centers - prev_centers)

                if prev_inertia is not None and (abs(prev_inertia - inertia) < self.tol or shift < self.tol):
                    break
                prev_inertia = inertia
                prev_centers = centers.copy()

            labels, dist_sq = self._assign_labels(X, centers)
            inertia = float(np.sum(dist_sq[np.arange(X.shape[0]), labels]))
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()
                best_iter = it

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_iter
        return self

    def predict(self, X: ArrayLike) -> np.ndarray | np.generic:
        if self.cluster_centers_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        if X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError("X has a different number of features than the fitted data")
        labels, _ = self._assign_labels(X, self.cluster_centers_)
        return labels[0] if labels.shape[0] == 1 else labels

    def transform(self, X: ArrayLike) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        if X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError("X has a different number of features than the fitted data")
        return np.sqrt(_squared_euclidean_distances(X, self.cluster_centers_))

    def fit_predict(self, X: ArrayLike, y: ArrayLike | None = None) -> np.ndarray:
        labels = self.fit(X, y).labels_
        if labels is None:
            raise RuntimeError("Model did not produce cluster labels")
        return labels
