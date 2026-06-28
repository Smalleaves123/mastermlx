from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils import as_2d, check_2d_array


class FastICA(BaseTransformer):
    """FastICA with symmetric decorrelation."""

    def __init__(self, n_components=None, max_iter=500, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.mean_ = None
        self.whitening_ = None
        self.unmixing_ = None
        self.mixing_ = None
        self.components_ = None
        self.n_iter_ = 0

    def _sym_decorrelation(self, W):
        vals, vecs = np.linalg.eigh(W @ W.T)
        vals = np.maximum(vals, 1e-12)
        return (vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T) @ W

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        n_samples, n_features = X.shape
        if self.n_components is None:
            n_comp = n_features
        else:
            n_comp = int(self.n_components)
            if n_comp < 1 or n_comp > n_features:
                raise ValueError("n_components must be between 1 and n_features")

        self.mean_ = np.mean(X, axis=0)
        Xc = X - self.mean_
        cov = np.cov(Xc, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1][:n_comp]
        vals = np.maximum(vals[order], 1e-12)
        vecs = vecs[:, order]

        self.whitening_ = vecs / np.sqrt(vals)[None, :]
        Xw = Xc @ self.whitening_

        rng = np.random.default_rng(self.random_state)
        W = rng.normal(size=(n_comp, n_comp))
        W = self._sym_decorrelation(W)

        for it in range(1, self.max_iter + 1):
            WX = Xw @ W.T
            gwx = np.tanh(WX)
            gprime = 1.0 - gwx ** 2
            W_new = (gwx.T @ Xw) / n_samples - np.diag(np.mean(gprime, axis=0)) @ W
            W_new = self._sym_decorrelation(W_new)

            lim = np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1.0))
            W = W_new
            if lim < self.tol:
                self.n_iter_ = it
                break
        else:
            self.n_iter_ = self.max_iter

        self.unmixing_ = W
        self.components_ = W @ self.whitening_.T
        self.mixing_ = np.linalg.pinv(self.components_)
        return self

    def transform(self, X):
        X = check_2d_array(X).astype(float)
        if self.components_ is None:
            raise RuntimeError("FastICA has not been fit yet")
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, X):
        X = as_2d(X).astype(float)
        if self.mixing_ is None:
            raise RuntimeError("FastICA has not been fit yet")
        return X @ self.mixing_.T + self.mean_


ICA = FastICA
