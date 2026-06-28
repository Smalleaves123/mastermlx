from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import accuracy, as_2d, check_1d_array, check_2d_array
from ._base import check_metric, check_weights, distance_weights, pairwise_neighbor_distance


class KNNClassifier(BaseEstimator):
    """k-nearest neighbors classifier."""

    def __init__(self, k=5, metric="euclidean", weights="uniform"):
        self.k = k
        self.metric = metric
        self.weights = weights
        self.X_ = None
        self.y_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if self.k < 1:
            raise ValueError("k must be at least 1")
        if self.k > X.shape[0]:
            raise ValueError("k cannot be larger than the number of training samples")
        check_metric(self.metric)
        check_weights(self.weights)

        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        return self

    def _dist(self, X):
        return pairwise_neighbor_distance(X, self.X_, self.metric)

    def predict(self, X):
        if self.X_ is None:
            raise RuntimeError("Model has not been fit yet")

        X = as_2d(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("X has a different number of features than the fitted data")

        dist = self._dist(X)
        nn = np.argsort(dist, axis=1)[:, : self.k]
        pred = []
        for i, row in enumerate(nn):
            labels = self.y_[row]
            vals = np.unique(labels)
            if self.weights == "uniform":
                cnt = np.array([np.sum(labels == label) for label in vals], dtype=float)
            else:
                w = distance_weights(dist[i, row])
                cnt = np.array([np.sum(w[labels == label]) for label in vals], dtype=float)
            pred.append(vals[int(np.argmax(cnt))])
        pred = np.asarray(pred)
        return pred[0] if pred.shape[0] == 1 else pred

    def score(self, X, y):
        return accuracy(y, self.predict(X))
