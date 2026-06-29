from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils.validation import check_2d_array


class CCA(BaseEstimator):
    """Canonical Correlation Analysis — finds linear combinations of two views
    with maximum correlation."""

    def __init__(self, n_components=2):
        self.n_components = int(n_components)
        self.x_weights_ = None
        self.y_weights_ = None
        self.corrs_ = None

    def fit(self, X, Y=None):
        if Y is None:
            raise ValueError("CCA requires two datasets X and Y")
        X = check_2d_array(X).astype(float)
        Y = check_2d_array(Y).astype(float)
        n = X.shape[0]
        if Y.shape[0] != n:
            raise ValueError("X and Y must have the same number of rows")
        k = min(self.n_components, X.shape[1], Y.shape[1])

        # Center
        Xc = X - X.mean(axis=0)
        Yc = Y - Y.mean(axis=0)

        # Regularized cross-covariance
        Cxx = (Xc.T @ Xc) / (n - 1) + 1e-8 * np.eye(X.shape[1])
        Cyy = (Yc.T @ Yc) / (n - 1) + 1e-8 * np.eye(Y.shape[1])
        Cxy = (Xc.T @ Yc) / (n - 1)

        # Solve generalized eigenvalue problem via SVD
        Lx = np.linalg.cholesky(Cxx)
        Ly = np.linalg.cholesky(Cyy)
        M = np.linalg.solve(Lx, Cxy) @ np.linalg.inv(Ly.T)
        U, s, Vt = np.linalg.svd(M, full_matrices=False)

        self.x_weights_ = np.linalg.solve(Lx.T, U[:, :k])
        self.y_weights_ = np.linalg.solve(Ly.T, Vt.T[:, :k])
        self.corrs_ = s[:k]
        return self

    def transform(self, X):
        X = check_2d_array(X).astype(float)
        if self.x_weights_ is None:
            raise RuntimeError("not fitted")
        Xc = X - X.mean(axis=0)
        return Xc @ self.x_weights_

    def fit_transform(self, X, Y=None):
        return self.fit(X, Y).transform(X)
