from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils.validation import check_2d_array


class KNNImputer(BaseTransformer):
    """Impute missing values using k-nearest neighbors."""

    def __init__(self, n_neighbors=5, weights="distance"):
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.X_fit_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        self.X_fit_ = X.copy()
        return self

    def transform(self, X):
        X = check_2d_array(X).astype(float)
        if self.X_fit_ is None:
            self.X_fit_ = X.copy()
        X_out = X.copy()
        n, d = X.shape

        for j in range(d):
            missing = np.isnan(X[:, j])
            if not np.any(missing):
                continue
            # Use columns that are complete for distance computation
            complete = ~np.any(np.isnan(X), axis=0)
            if np.sum(complete) < 2:
                continue
            X_complete = X[:, complete]
            X_fit_complete = self.X_fit_[:, complete]

            for i in np.flatnonzero(missing):
                # Distance to all rows that have this feature non-NaN
                diff = X_fit_complete - X_complete[i]
                dist = np.sqrt(np.sum(diff**2, axis=1))
                dist[i] = np.inf  # exclude self
                nn = np.argsort(dist)[:self.n_neighbors]

                if self.weights == "uniform":
                    X_out[i, j] = np.nanmean(self.X_fit_[nn, j])
                else:
                    w = 1.0 / np.maximum(dist[nn], 1e-12)
                    w_sum = np.sum(w)
                    X_out[i, j] = np.sum(w * self.X_fit_[nn, j]) / max(w_sum, 1e-12)
        return X_out

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
