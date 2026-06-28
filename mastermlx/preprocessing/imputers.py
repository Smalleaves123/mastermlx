from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils.validation import check_2d_array


class SimpleImputer(BaseTransformer):
    """Fill missing values with a per-column statistic."""

    def __init__(self, strategy="mean", fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        if self.strategy not in {"mean", "median", "most_frequent", "constant"}:
            raise ValueError("strategy must be one of: mean, median, most_frequent, constant")

        stats = []
        for j in range(X.shape[1]):
            col = X[:, j]
            valid = col[~np.isnan(col)]
            if self.strategy == "mean":
                value = float(np.mean(valid)) if valid.size else 0.0
            elif self.strategy == "median":
                value = float(np.median(valid)) if valid.size else 0.0
            elif self.strategy == "most_frequent":
                if valid.size == 0:
                    value = 0.0
                else:
                    vals, cnt = np.unique(valid, return_counts=True)
                    value = float(vals[np.argmax(cnt)])
            else:
                value = 0.0 if self.fill_value is None else float(self.fill_value)
            stats.append(value)
        self.statistics_ = np.asarray(stats, dtype=float)
        return self

    def transform(self, X):
        X = check_2d_array(X).astype(float)
        if self.statistics_ is None:
            raise RuntimeError("Imputer has not been fit yet")
        if X.shape[1] != self.statistics_.shape[0]:
            raise ValueError("X has a different number of features than the fitted data")
        out = X.copy()
        for j in range(out.shape[1]):
            mask = np.isnan(out[:, j])
            out[mask, j] = self.statistics_[j]
        return out
