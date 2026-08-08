from __future__ import annotations

import numpy as np
from typing import cast

from ..base import BaseEstimator
from ..accel.ml_kernels import radius_neighbors
from ..utils import as_2d, check_1d_array, check_2d_array, r2_score
from ._base import check_metric, check_weights, distance_weights, knn_neighbors, pairwise_neighbor_distance


class RadiusNeighborsRegressor(BaseEstimator):
    """Radius neighbors regressor."""

    def __init__(self, radius=1.0, metric="euclidean", weights="uniform"):
        self.radius = float(radius)
        self.metric = metric
        self.weights = weights
        self.X_ = None
        self.y_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be positive and finite")
        check_metric(self.metric)
        check_weights(self.weights)
        self.X_ = X
        self.y_ = y.astype(float)
        self._set_n_features(X)
        return self

    def predict(self, X):
        if self.X_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        y_train = cast(np.ndarray, self.y_)
        pred = np.zeros(X.shape[0], dtype=float)
        if self.metric == "euclidean":
            indptr, indices, distances = radius_neighbors(X, self.X_, self.radius)
            empty = indptr[1:] == indptr[:-1]
            nearest = knn_neighbors(X, self.X_, 1, self.metric) if np.any(empty) else None
            for i in range(X.shape[0]):
                start, end = int(indptr[i]), int(indptr[i + 1])
                if start == end:
                    if nearest is None:
                        raise RuntimeError("nearest-neighbor fallback was not initialized")
                    nearest_indices, _ = cast(tuple[np.ndarray, np.ndarray], nearest)
                    pred[i] = y_train[nearest_indices[i, 0]]
                    continue
                row_indices = indices[start:end]
                vals = y_train[row_indices]
                if self.weights == "uniform":
                    pred[i] = np.mean(vals)
                else:
                    w = distance_weights(distances[start:end])
                    pred[i] = np.sum(w * vals) / np.sum(w)
        else:
            dist = pairwise_neighbor_distance(X, cast(np.ndarray, self.X_), self.metric)
            for i in range(X.shape[0]):
                mask = dist[i] <= self.radius
                if not np.any(mask):
                    nearest = int(np.argmin(dist[i]))
                    pred[i] = y_train[nearest]
                    continue
                vals = y_train[mask]
                if self.weights == "uniform":
                    pred[i] = np.mean(vals)
                else:
                    w = distance_weights(dist[i, mask])
                    pred[i] = np.sum(w * vals) / np.sum(w)
        return pred

    def score(self, X, y):
        return r2_score(y, self.predict(X))
