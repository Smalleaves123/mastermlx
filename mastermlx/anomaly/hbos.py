from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import as_2d, check_2d_array


class HBOS(BaseEstimator):
    """Histogram-based outlier score."""

    def __init__(self, n_bins=10, contamination=0.1, alpha=1e-6):
        self.n_bins = int(n_bins)
        self.contamination = float(contamination)
        self.alpha = float(alpha)
        self.bin_edges_ = None
        self.bin_density_ = None
        self.offset_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        n_samples, n_features = X.shape
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        if not 0.0 < self.contamination < 0.5:
            raise ValueError("contamination must be in (0, 0.5)")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")

        edges = []
        dens = []
        for j in range(n_features):
            counts, cur_edges = np.histogram(X[:, j], bins=self.n_bins, density=False)
            widths = np.diff(cur_edges)
            prob = counts / max(n_samples, 1)
            cur_dens = prob / np.maximum(widths, self.alpha)
            cur_dens = np.maximum(cur_dens, self.alpha)
            edges.append(cur_edges)
            dens.append(cur_dens)

        self.bin_edges_ = edges
        self.bin_density_ = dens
        scores = self.score_samples(X)
        self.offset_ = float(np.quantile(scores, 1.0 - self.contamination))
        return self

    def score_samples(self, X):
        if self.bin_edges_ is None or self.bin_density_ is None:
            raise RuntimeError("HBOS has not been fit yet")
        X = as_2d(X).astype(float)
        if X.shape[1] != len(self.bin_edges_):
            raise ValueError("X has a different number of features than the fitted data")

        scores = np.zeros(X.shape[0], dtype=float)
        for j, (edges, dens) in enumerate(zip(self.bin_edges_, self.bin_density_)):
            idx = np.searchsorted(edges, X[:, j], side="right") - 1
            idx = np.clip(idx, 0, dens.shape[0] - 1)
            scores += -np.log(dens[idx])
        return float(scores[0]) if scores.shape[0] == 1 else scores

    def decision_function(self, X):
        scores = self.score_samples(X)
        out = self.offset_ - scores
        return float(out) if np.ndim(out) == 0 else out

    def predict(self, X):
        scores = self.score_samples(X)
        pred = np.where(scores >= self.offset_, -1, 1)
        return int(pred) if np.ndim(pred) == 0 else pred
