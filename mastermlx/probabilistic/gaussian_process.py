from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import as_2d, check_1d_array, check_2d_array, check_same_rows, r2_score


def _rbf_kernel(X, Y, length_scale):
    x2 = np.sum(X ** 2, axis=1)[:, None]
    y2 = np.sum(Y ** 2, axis=1)[None, :]
    d2 = np.maximum(x2 + y2 - 2.0 * (X @ Y.T), 0.0)
    return np.exp(-0.5 * d2 / (length_scale ** 2))


class GaussianProcessRegressor(BaseEstimator):
    """Gaussian process regressor with RBF kernel."""

    def __init__(self, length_scale=1.0, alpha=1e-6):
        self.length_scale = float(length_scale)
        self.alpha = float(alpha)
        self.X_train_ = None
        self.y_train_ = None
        self.L_ = None
        self.alpha_vec_ = None

    def _posterior(self, X):
        X = as_2d(X).astype(float)
        if self.X_train_ is None or self.alpha_vec_ is None or self.L_ is None:
            raise RuntimeError("Model has not been fit yet")
        if X.shape[1] != self.X_train_.shape[1]:
            raise ValueError("X has a different number of features than the fitted data")

        K_trans = _rbf_kernel(X, self.X_train_, self.length_scale)
        mean = K_trans @ self.alpha_vec_
        v = np.linalg.solve(self.L_, K_trans.T)
        K_xx = _rbf_kernel(X, X, self.length_scale)
        cov = K_xx - v.T @ v
        cov = 0.5 * (cov + cov.T)
        return mean, cov

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y).astype(float)
        X, y = check_same_rows(X, y)
        if self.length_scale <= 0.0:
            raise ValueError("length_scale must be positive")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")

        K = _rbf_kernel(X, X, self.length_scale)
        K = K + self.alpha * np.eye(X.shape[0])
        self.L_ = np.linalg.cholesky(K)
        self.alpha_vec_ = np.linalg.solve(self.L_.T, np.linalg.solve(self.L_, y))
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X, return_std=False):
        mean, cov = self._posterior(X)
        if not return_std:
            return float(mean[0]) if mean.shape[0] == 1 else mean

        var = np.maximum(np.diag(cov), 0.0)
        std = np.sqrt(var)
        if mean.shape[0] == 1:
            return float(mean[0]), float(std[0])
        return mean, std

    def posterior_summary(self):
        if self.X_train_ is None or self.alpha_vec_ is None:
            raise RuntimeError("Model has not been fit yet")
        return {
            "model": self.__class__.__name__,
            "length_scale": float(self.length_scale),
            "alpha": float(self.alpha),
            "n_train": int(self.X_train_.shape[0]),
            "n_features": int(self.X_train_.shape[1]),
        }

    def sample_posterior_functions(self, X, n_samples=1, random_state=None):
        mean, cov = self._posterior(X)
        jitter = 1e-10 * np.eye(cov.shape[0])
        rng = np.random.default_rng(random_state)
        samples = rng.multivariate_normal(mean, cov + jitter, size=int(n_samples))
        if int(n_samples) == 1:
            return float(samples[0, 0]) if samples.shape[1] == 1 else samples[0]
        return samples[:, 0] if samples.shape[1] == 1 else samples

    def score(self, X, y):
        return r2_score(y, self.predict(X))


GPR = GaussianProcessRegressor
