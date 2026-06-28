from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils import as_2d, check_2d_array
from ..utils.kernels import pairwise_kernel, resolve_gamma


class KernelPCA(BaseTransformer):
    """Kernel PCA with a selection of pairwise kernels."""

    def __init__(self, n_components=None, kernel="rbf", gamma=None, degree=3, coef0=1.0):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.X_fit_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.row_mean_ = None
        self.total_mean_ = None

    def _kernel(self, X, Y):
        return pairwise_kernel(X, Y, kernel=self.kernel, gamma=self._gamma, coef0=self.coef0, degree=self.degree)

    def _center_fit_kernel(self, K):
        self.row_mean_ = np.mean(K, axis=0)
        self.total_mean_ = float(np.mean(K))
        return K - self.row_mean_[None, :] - self.row_mean_[:, None] + self.total_mean_

    def _center_new_kernel(self, K):
        row_mean = np.mean(K, axis=1, keepdims=True)
        return K - self.row_mean_[None, :] - row_mean + self.total_mean_

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        n_samples, n_features = X.shape
        if self.n_components is None:
            k = n_samples
        else:
            k = int(self.n_components)
            if k < 1 or k > n_samples:
                raise ValueError("n_components must be between 1 and n_samples")

        self._gamma = resolve_gamma(self.gamma, n_features)
        self.X_fit_ = X

        K = self._kernel(X, X)
        Kc = self._center_fit_kernel(K)
        vals, vecs = np.linalg.eigh(Kc)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        keep = vals > 1e-12
        vals = vals[keep][:k]
        vecs = vecs[:, keep][:, :k]
        if vals.size == 0:
            raise ValueError("Kernel matrix is numerically rank deficient")

        self.eigenvalues_ = vals
        self.eigenvectors_ = vecs / np.sqrt(vals)[None, :]
        return self

    def transform(self, X):
        X = as_2d(X).astype(float)
        if self.eigenvectors_ is None:
            raise RuntimeError("KernelPCA has not been fit yet")
        K = self._kernel(X, self.X_fit_)
        Kc = self._center_new_kernel(K)
        return Kc @ self.eigenvectors_

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        K = self._kernel(self.X_fit_, self.X_fit_)
        Kc = self._center_new_kernel(K)
        return Kc @ self.eigenvectors_


KPCA = KernelPCA
