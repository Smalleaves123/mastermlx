from __future__ import annotations

import numpy as np
from typing import cast

from ..base import BaseEstimator
from ..utils import as_2d, check_1d_array, check_2d_array, mean_squared_error
from ._base import check_metric, check_weights, distance_weights, pairwise_neighbor_distance


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
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        check_metric(self.metric)
        check_weights(self.weights)
        self.X_ = X
        self.y_ = y.astype(float)
        return self

    def predict(self, X):
        if self.X_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        y_train = cast(np.ndarray, self.y_)
        dist = pairwise_neighbor_distance(X, cast(np.ndarray, self.X_), self.metric)
        pred = np.zeros(X.shape[0], dtype=float)
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
        return float(pred[0]) if pred.shape[0] == 1 else pred

    def score(self, X, y):
        return -mean_squared_error(y, self.predict(X))
