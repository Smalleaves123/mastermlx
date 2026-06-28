from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils import as_2d, check_2d_array


class TruncatedSVD(BaseTransformer):
    """Truncated SVD without centering the input matrix."""

    def __init__(self, n_components=2):
        self.n_components = int(n_components)
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        n_samples, n_features = X.shape
        if self.n_components < 1 or self.n_components > min(n_samples, n_features):
            raise ValueError("n_components must be between 1 and min(n_samples, n_features)")

        U, s, vt = np.linalg.svd(X, full_matrices=False)
        k = self.n_components
        self.components_ = vt[:k]
        self.singular_values_ = s[:k]

        denom = max(1, n_samples - 1)
        var = (s ** 2) / denom
        self.explained_variance_ = var[:k]
        total = np.var(X, axis=0, ddof=1).sum()
        if total <= 0.0:
            self.explained_variance_ratio_ = np.zeros(k)
        else:
            self.explained_variance_ratio_ = self.explained_variance_ / total
        return self

    def transform(self, X):
        X = check_2d_array(X).astype(float)
        if self.components_ is None:
            raise RuntimeError("TruncatedSVD has not been fit yet")
        if X.shape[1] != self.components_.shape[1]:
            raise ValueError("X has a different number of features than the fitted data")
        return X @ self.components_.T

    def inverse_transform(self, X):
        X = as_2d(X).astype(float)
        if self.components_ is None:
            raise RuntimeError("TruncatedSVD has not been fit yet")
        return X @ self.components_


TSVD = TruncatedSVD
