from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils.validation import check_2d_array


class Normalizer(BaseTransformer):
    """Normalize samples to unit norm."""

    def __init__(self, norm="l2"):
        self.norm = norm

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        if self.norm not in {"l1", "l2", "max"}:
            raise ValueError("norm must be one of: l1, l2, max")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = check_2d_array(X).astype(float)
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("Normalizer has not been fit yet")
        if self.norm == "l1":
            scale = np.sum(np.abs(X), axis=1, keepdims=True)
        elif self.norm == "l2":
            scale = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
        else:
            scale = np.max(np.abs(X), axis=1, keepdims=True)
        scale = np.where(scale == 0.0, 1.0, scale)
        return X / scale
