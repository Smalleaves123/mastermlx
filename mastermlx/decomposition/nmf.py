from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils import check_2d_array


class NMF(BaseTransformer):
    """Non-negative matrix factorization with multiplicative updates."""

    def __init__(self, n_components, max_iter=500, tol=1e-4, random_state=None):
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.components_ = None
        self.reconstruction_err_ = None
        self.n_iter_ = 0

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        if np.any(X < 0):
            raise ValueError("NMF requires non-negative input data")
        n_samples, n_features = X.shape
        if self.n_components < 1 or self.n_components > min(n_samples, n_features):
            raise ValueError("n_components must be between 1 and min(n_samples, n_features)")

        rng = np.random.default_rng(self.random_state)
        W = rng.random((n_samples, self.n_components)) + 1e-6
        H = rng.random((self.n_components, n_features)) + 1e-6
        prev_err = None

        for it in range(1, self.max_iter + 1):
            WH = W @ H
            H *= (W.T @ X) / np.maximum(W.T @ WH, 1e-12)
            WH = W @ H
            W *= (X @ H.T) / np.maximum(WH @ H.T, 1e-12)

            err = float(np.linalg.norm(X - W @ H, ord="fro"))
            if prev_err is not None and abs(prev_err - err) < self.tol:
                self.n_iter_ = it
                break
            prev_err = err
        else:
            self.n_iter_ = self.max_iter

        self.components_ = H
        self.W_ = W
        self.reconstruction_err_ = prev_err
        return self

    def transform(self, X):
        X = check_2d_array(X).astype(float)
        if self.components_ is None:
            raise RuntimeError("NMF has not been fit yet")
        if np.any(X < 0):
            raise ValueError("NMF requires non-negative input data")
        rng = np.random.default_rng(self.random_state)
        W = rng.random((X.shape[0], self.n_components)) + 1e-6
        H = self.components_
        for _ in range(max(50, self.n_iter_)):
            WH = W @ H
            W *= (X @ H.T) / np.maximum(WH @ H.T, 1e-12)
        return W

    def inverse_transform(self, W):
        W = check_2d_array(W).astype(float)
        if self.components_ is None:
            raise RuntimeError("NMF has not been fit yet")
        if W.shape[1] != self.components_.shape[0]:
            raise ValueError("W has a different number of components than the fitted model")
        return W @ self.components_
