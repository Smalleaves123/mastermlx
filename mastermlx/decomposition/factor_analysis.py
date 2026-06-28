from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils import as_2d, check_2d_array


class FactorAnalysis(BaseTransformer):
    """Factor analysis with diagonal noise covariance."""

    def __init__(self, n_components=2):
        self.n_components = int(n_components)
        self.mean_ = None
        self.components_ = None
        self.noise_variance_ = None
        self._proj_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        n_samples, n_features = X.shape
        if self.n_components < 1 or self.n_components > n_features:
            raise ValueError("n_components must be between 1 and n_features")

        self.mean_ = np.mean(X, axis=0)
        Xc = X - self.mean_
        cov = np.cov(Xc, rowvar=False)

        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        k = self.n_components
        main_vals = np.maximum(vals[:k], 1e-12)
        if k < n_features:
            psi0 = float(np.mean(np.maximum(vals[k:], 1e-12)))
        else:
            psi0 = 1e-6

        load = vecs[:, :k] * np.sqrt(np.maximum(main_vals - psi0, 1e-12))[None, :]
        noise = np.diag(cov - load @ load.T)
        noise = np.maximum(noise, 1e-12)

        psi_inv = 1.0 / noise
        mid = np.eye(k) + load.T @ (psi_inv[:, None] * load)
        self._proj_ = (psi_inv[:, None] * load) @ np.linalg.inv(mid)

        self.components_ = load.T
        self.noise_variance_ = noise
        return self

    def transform(self, X):
        X = check_2d_array(X).astype(float)
        if self.components_ is None:
            raise RuntimeError("FactorAnalysis has not been fit yet")
        return (X - self.mean_) @ self._proj_

    def inverse_transform(self, X):
        X = as_2d(X).astype(float)
        if self.components_ is None:
            raise RuntimeError("FactorAnalysis has not been fit yet")
        return X @ self.components_ + self.mean_
