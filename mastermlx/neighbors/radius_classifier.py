from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import accuracy, as_2d, check_1d_array, check_2d_array
from ._base import check_metric, check_weights, distance_weights, pairwise_neighbor_distance


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
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        check_metric(self.metric)
        check_weights(self.weights)
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        if self.X_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        dist = pairwise_neighbor_distance(X, self.X_, self.metric)
        pred = []
        for i in range(X.shape[0]):
            mask = dist[i] <= self.radius
            if not np.any(mask):
                nearest = int(np.argmin(dist[i]))
                pred.append(self.y_[nearest])
                continue
            labels = self.y_[mask]
            vals = np.unique(labels)
            if self.weights == "uniform":
                cnt = np.array([np.sum(labels == label) for label in vals], dtype=float)
            else:
                w = distance_weights(dist[i, mask])
                cnt = np.array([np.sum(w[labels == label]) for label in vals], dtype=float)
            pred.append(vals[int(np.argmax(cnt))])
        pred = np.asarray(pred)
        return pred[0] if pred.shape[0] == 1 else pred

    def score(self, X, y):
        return accuracy(y, self.predict(X))
