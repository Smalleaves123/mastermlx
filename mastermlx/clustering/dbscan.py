from __future__ import annotations

import numpy as np

from ..accel.ml_kernels import dbscan_labels, dbscan_neighbors
from ..base import BaseEstimator
from ..utils import check_2d_array


class DBSCAN(BaseEstimator):
    """Density-based clustering with an optional C++ acceleration path."""

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.labels_ = None
        self.core_sample_indices_ = None
        self.components_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X)
        if not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be positive and finite")
        if self.min_samples < 1:
            raise ValueError("min_samples must be at least 1")

        indptr, indices, _ = dbscan_neighbors(X, self.eps)
        labels, core_samples = dbscan_labels(indptr, indices, self.min_samples)
        cluster_id = int(np.max(labels)) + 1 if np.any(labels >= 0) else 0

        self.labels_ = labels
        self.core_sample_indices_ = core_samples
        self.components_ = X[core_samples]
        self.n_clusters_ = cluster_id
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_

    def predict(self, X):
        if self.labels_ is None:
            raise RuntimeError("Model has not been fit yet")
        raise NotImplementedError("DBSCAN does not support predicting labels for new samples")
