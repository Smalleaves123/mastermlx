from __future__ import annotations

import numpy as np
from typing import cast

from ..base import BaseTransformer
from ..utils import as_2d, check_2d_array

try:
    from scipy.linalg import svd as _scipy_svd
except ImportError:  # pragma: no cover - SciPy is an optional comparison dependency
    _scipy_svd = None


def _compute_svd(X):
    """Use SciPy's LAPACK driver when available, with a NumPy-only fallback."""

    if X.shape[0] >= 10 * X.shape[1] and X.shape[1] < 1000:
        covariance = X.T @ X
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        return None, np.sqrt(eigenvalues), eigenvectors[:, order].T

    if _scipy_svd is not None:
        return _scipy_svd(
            X,
            full_matrices=False,
            lapack_driver="gesdd",
            check_finite=False,
            overwrite_a=True,
        )
    return np.linalg.svd(X, full_matrices=False)


class PCA(BaseTransformer):
    """Principal component analysis using SVD."""

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X, y=None):
        X = np.ascontiguousarray(check_2d_array(X), dtype=float)
        self._set_n_features(X)
        n, m = X.shape
        if self.n_components is None:
            k = m
        else:
            k = int(self.n_components)
            if k < 1 or k > m:
                raise ValueError("n_components must be between 1 and n_features")

        self.mean_ = np.mean(X, axis=0)
        Xc = X - self.mean_
        _, s, vt = _compute_svd(Xc)

        self.components_ = vt[:k]
        denom = max(1, n - 1)
        var = (s ** 2) / denom
        self.explained_variance_ = var[:k]

        total = var.sum()
        if total == 0:
            self.explained_variance_ratio_ = np.zeros(k)
        else:
            self.explained_variance_ratio_ = self.explained_variance_ / total

        return self

    def transform(self, X):
        self._check_fitted(["components_", "mean_"])
        X = self._check_X(X)
        return (X - cast(np.ndarray, self.mean_)) @ cast(np.ndarray, self.components_).T

    def inverse_transform(self, X):
        self._check_fitted(["components_", "mean_"])
        X = as_2d(X)
        return X @ cast(np.ndarray, self.components_) + cast(np.ndarray, self.mean_)


PC = PCA
