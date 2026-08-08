from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import accuracy, as_2d, check_1d_array, check_2d_array
from ._base import check_metric, check_weights, distance_weights, knn_neighbors


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
        self._set_n_features(X)
        self.classes_, self.y_codes_ = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]
        return self

    def predict(self, X):
        if self.X_ is None:
            raise RuntimeError("Model has not been fit yet")

        X = as_2d(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("X has a different number of features than the fitted data")

        nn, dist = knn_neighbors(X, self.X_, self.k, self.metric)
        pred_codes = np.empty(X.shape[0], dtype=int)

        for i, row in enumerate(nn):
            codes = self.y_codes_[row]
            if self.weights == "uniform":
                cnt = np.bincount(codes, minlength=self.n_classes_)
            else:
                w = distance_weights(dist[i])
                cnt = np.bincount(codes, weights=w, minlength=self.n_classes_)
            pred_codes[i] = int(np.argmax(cnt))

        return self.classes_[pred_codes]

    def score(self, X, y):
        return accuracy(y, self.predict(X))
