from __future__ import annotations

import numpy as np
from typing import cast

from ..base import BaseEstimator
from ..accel.ml_kernels import radius_neighbors
from ..utils import accuracy, as_2d, check_1d_array, check_2d_array
from ._base import check_metric, check_weights, distance_weights, knn_neighbors, pairwise_neighbor_distance


class RadiusNeighborsClassifier(BaseEstimator):
    """Radius neighbors classifier."""

    def __init__(self, radius=1.0, metric="euclidean", weights="uniform"):
        self.radius = float(radius)
        self.metric = metric
        self.weights = weights
        self.X_ = None
        self.y_ = None
        self.classes_ = None

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
        self.classes_, self.y_codes_ = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]
        return self

    def predict(self, X):
        if self.X_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        y_codes = cast(np.ndarray, self.y_codes_)
        classes = cast(np.ndarray, self.classes_)
        pred_codes = np.empty(X.shape[0], dtype=int)
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
                    pred_codes[i] = int(y_codes[nearest_indices[i, 0]])
                    continue
                row_indices = indices[start:end]
                codes = y_codes[row_indices]
                row_distances = distances[start:end]
                if self.weights == "uniform":
                    cnt = np.bincount(codes, minlength=self.n_classes_)
                else:
                    w = distance_weights(row_distances)
                    cnt = np.bincount(codes, weights=w, minlength=self.n_classes_)
                pred_codes[i] = int(np.argmax(cnt))
        else:
            dist = pairwise_neighbor_distance(X, self.X_, self.metric)
            for i in range(X.shape[0]):
                mask = dist[i] <= self.radius
                if not np.any(mask):
                    nearest = int(np.argmin(dist[i]))
                    pred_codes[i] = int(y_codes[nearest])
                    continue
                codes = y_codes[mask]
                if self.weights == "uniform":
                    cnt = np.bincount(codes, minlength=self.n_classes_)
                else:
                    w = distance_weights(dist[i, mask])
                    cnt = np.bincount(codes, weights=w, minlength=self.n_classes_)
                pred_codes[i] = int(np.argmax(cnt))
        pred = classes[pred_codes]
        return pred[0] if pred.shape[0] == 1 else pred

    def score(self, X, y):
        return accuracy(y, self.predict(X))
