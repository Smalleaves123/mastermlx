from __future__ import annotations

import numpy as np
from typing import cast

from ..base import BaseEstimator
from ..utils import accuracy, as_2d, check_1d_array, check_2d_array
from ..utils.distance import pairwise_distance


class NearestCentroid(BaseEstimator):
    """Nearest centroid classifier — classify by closest class mean."""

    def __init__(self, metric="euclidean"):
        self.metric = metric
        self.centroids_ = None
        self.classes_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows")
        self.classes_ = np.unique(y)
        self.centroids_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X):
        X = as_2d(X).astype(float)
        if self.centroids_ is None:
            raise RuntimeError("not fitted")
        dists = pairwise_distance(X, self.centroids_, metric=self.metric)
        idx = np.argmin(dists, axis=1)
        return cast(np.ndarray, self.classes_)[idx]

    def score(self, X, y):
        return accuracy(y, self.predict(X))
