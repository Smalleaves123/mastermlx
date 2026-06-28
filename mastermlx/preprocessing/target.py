from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils.validation import check_2d_array, check_1d_array, check_same_rows


class TargetEncoder(BaseTransformer):
    """Replace each category with the mean target value, with smoothing."""

    def __init__(self, smoothing=1.0):
        self.smoothing = float(smoothing)
        self.maps_ = {}
        self.global_mean_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y).astype(float)
        X, y = check_same_rows(X, y)
        self.global_mean_ = float(np.mean(y))
        self.maps_ = {}
        for j in range(X.shape[1]):
            col_vals = X[:, j]
            uniq = np.unique(col_vals)
            enc = {}
            for v in uniq:
                mask = col_vals == v
                n = np.sum(mask)
                enc[v] = (np.sum(y[mask]) + self.smoothing * self.global_mean_) / (n + self.smoothing)
            self.maps_[j] = enc
        return self

    def transform(self, X):
        X = check_2d_array(X)
        if not self.maps_:
            raise RuntimeError("Encoder has not been fit yet")
        out = np.full(X.shape, self.global_mean_, dtype=float)
        for j in range(X.shape[1]):
            enc = self.maps_.get(j, {})
            for i in range(X.shape[0]):
                out[i, j] = enc.get(X[i, j], self.global_mean_)
        return out
