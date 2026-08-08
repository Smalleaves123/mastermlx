from __future__ import annotations

import numpy as np
from typing import cast

from ..base import BaseEstimator
from ..accel.ml_kernels import gmm_log_gaussian, gmm_m_step
from ..utils import as_2d, check_2d_array
from ..utils.math import log_sum_exp


class GMM(BaseEstimator):
    """Gaussian mixture model trained with EM."""

    def __init__(self, n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.resp_ = None
        self.lower_bound_ = []

    def _log_gaussian_components(self, X):
        means = cast(np.ndarray, self.means_)
        covariances = cast(np.ndarray, self.covariances_)
        precisions = np.empty_like(covariances)
        log_determinants = np.empty(means.shape[0], dtype=float)
        for j, cov in enumerate(covariances):
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                raise ValueError("Covariance matrix must be positive definite")
            precisions[j] = np.linalg.inv(cov)
            log_determinants[j] = logdet
        return gmm_log_gaussian(X, means, precisions, log_determinants)

    def fit(self, X, y=None):
        X = check_2d_array(X)
        n, d = X.shape
        k = int(self.n_components)
        if k < 1 or k > n:
            raise ValueError("n_components must be between 1 and number of samples")

        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n, size=k, replace=False)
        self.means_ = X[idx].copy()
        self.weights_ = np.full(k, 1.0 / k)
        base_cov = np.cov(X, rowvar=False)
        if base_cov.ndim == 0:
            base_cov = np.array([[float(base_cov)]])
        self.covariances_ = np.repeat(base_cov[None, :, :], k, axis=0)
        self.covariances_ += self.reg_covar * np.eye(d)[None, :, :]

        prev = None
        self.lower_bound_ = []
        for _ in range(self.max_iter):
            log_prob = self._log_gaussian_components(X) + np.log(self.weights_ + 1e-12)[None, :]
            log_norm = log_sum_exp(log_prob, axis=1)
            resp = np.exp(log_prob - log_norm[:, None])

            self.weights_, self.means_, self.covariances_ = gmm_m_step(X, resp, self.reg_covar)
            self.resp_ = resp

            lb = np.mean(log_norm)
            self.lower_bound_.append(lb)
            if prev is not None and abs(lb - prev) < self.tol:
                break
            prev = lb

        return self

    def predict_proba(self, X):
        if self.means_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        log_prob = self._log_gaussian_components(X) + np.log(cast(np.ndarray, self.weights_) + 1e-12)[None, :]
        log_norm = log_sum_exp(log_prob, axis=1)
        resp = np.exp(log_prob - log_norm[:, None])
        return resp

    def predict(self, X):
        resp = self.predict_proba(X)
        idx = np.argmax(resp, axis=1)
        return idx

    def score(self, X, y=None):
        X = check_2d_array(X)
        log_prob = self._log_gaussian_components(X) + np.log(cast(np.ndarray, self.weights_) + 1e-12)[None, :]
        return float(np.mean(log_sum_exp(log_prob, axis=1)))
