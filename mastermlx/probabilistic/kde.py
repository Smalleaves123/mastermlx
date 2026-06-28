from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils.validation import check_2d_array


class KernelDensity(BaseEstimator):
    """Kernel Density Estimation with Gaussian/RBF kernel."""

    def __init__(self, bandwidth=1.0, kernel="gaussian"):
        self.bandwidth = float(bandwidth)
        self.kernel = kernel
        self.X_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        self.X_ = X
        return self

    def score_samples(self, X):
        X = check_2d_array(X).astype(float)
        if self.X_ is None:
            raise RuntimeError("Model has not been fit yet")
        n, d = self.X_.shape
        h = self.bandwidth
        if self.kernel == "gaussian":
            sq = np.sum(X**2, axis=1)[:, None] + np.sum(self.X_**2, axis=1)[None, :] \
                 - 2.0 * (X @ self.X_.T)
            log_kernel = -0.5 * sq / (h ** 2)
            log_norm = -0.5 * d * np.log(2.0 * np.pi) - d * np.log(h) - np.log(n)
            max_log = np.max(log_kernel, axis=1, keepdims=True)
            log_density = log_norm + max_log.ravel() + np.log(np.sum(np.exp(log_kernel - max_log), axis=1))
            return log_density.ravel()
        if self.kernel == "tophat":
            sq = np.sum((X[:, None] - self.X_[None, :])**2, axis=2)
            in_ball = (sq <= h**2).astype(float)
            vol = (h ** d) * np.pi ** (d / 2.0) / np.exp(np.log(np.pi) * d / 2.0 + _log_gamma(d / 2.0 + 1))
            dens = np.sum(in_ball, axis=1) / (n * max(vol, 1e-12))
            return np.log(np.maximum(dens, 1e-15))

        raise ValueError("kernel must be 'gaussian' or 'tophat'")

    def score(self, X, y=None):
        return float(np.mean(self.score_samples(X)))

    def sample(self, n_samples=1, random_state=None):
        if self.X_ is None:
            raise RuntimeError("Model has not been fit yet")
        rng = np.random.default_rng(random_state)
        idx = rng.integers(0, self.X_.shape[0], size=n_samples)
        return self.X_[idx] + rng.normal(0, self.bandwidth, size=(n_samples, self.X_.shape[1]))


def _log_gamma(x):
    from math import gamma, log
    return log(gamma(x))
