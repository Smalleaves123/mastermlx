from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..accel.ml_kernels import knn_impute
from ..utils.validation import check_2d_array


class KNNImputer(BaseTransformer):
    """Impute missing values using k-nearest neighbors."""

    def __init__(self, n_neighbors=5, weights="distance"):
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.X_fit_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        if np.isinf(X).any():
            raise ValueError("X must not contain infinite values")
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        if self.weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'")
        self.X_fit_ = X.copy()
        return self

    def transform(self, X):
        X = check_2d_array(X).astype(float)
        if self.X_fit_ is None:
            self.X_fit_ = X.copy()
        if np.isinf(X).any():
            raise ValueError("X must not contain infinite values")
        return knn_impute(X, self.X_fit_, self.n_neighbors, self.weights)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
