from __future__ import annotations

import numpy as np

from ..accel import pairwise_distances
from ..base import BaseEstimator
from ..utils import as_2d, check_2d_array


class LocalOutlierFactor(BaseEstimator):
    """Local outlier factor for density-based anomaly detection."""

    def __init__(self, n_neighbors=20, contamination=0.1):
        self.n_neighbors = int(n_neighbors)
        self.contamination = float(contamination)
        self.X_ = None
        self.k_distance_ = None
        self.lrd_ = None
        self.offset_ = None
        self.negative_outlier_factor_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        n_samples = X.shape[0]
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        if self.n_neighbors >= n_samples:
            raise ValueError("n_neighbors must be smaller than the number of samples")
        if not 0.0 < self.contamination < 0.5:
            raise ValueError("contamination must be in (0, 0.5)")

        self.X_ = X
        dist = pairwise_distances(X, X)
        np.fill_diagonal(dist, np.inf)
        nn = np.argsort(dist, axis=1)[:, : self.n_neighbors]
        d_sorted = np.take_along_axis(dist, nn, axis=1)
        self.k_distance_ = d_sorted[:, -1]

        reach = np.maximum(d_sorted, self.k_distance_[nn])
        self.lrd_ = 1.0 / (np.mean(reach, axis=1) + 1e-12)

        lof = np.mean(self.lrd_[nn] / self.lrd_[:, None], axis=1)
        self.negative_outlier_factor_ = -lof
        self.offset_ = float(np.quantile(lof, 1.0 - self.contamination))
        return self

    def score_samples(self, X):
        if self.X_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X).astype(float)
        dist = pairwise_distances(X, self.X_)
        nn = np.argsort(dist, axis=1)[:, : self.n_neighbors]
        d_sorted = np.take_along_axis(dist, nn, axis=1)
        reach = np.maximum(d_sorted, self.k_distance_[nn])
        lrd = 1.0 / (np.mean(reach, axis=1) + 1e-12)
        lof = np.mean(self.lrd_[nn] / lrd[:, None], axis=1)
        out = -lof
        return float(out[0]) if out.shape[0] == 1 else out

    def decision_function(self, X):
        scores = self.score_samples(X)
        out = scores + self.offset_
        return float(out) if np.ndim(out) == 0 else out

    def predict(self, X):
        scores = self.score_samples(X)
        pred = np.where(-scores > self.offset_, -1, 1)
        return int(pred) if np.ndim(pred) == 0 else pred
