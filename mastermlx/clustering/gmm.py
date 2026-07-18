from __future__ import annotations

import numpy as np
from typing import cast

from ..base import BaseEstimator
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

    def _log_gauss(self, X, mean, cov):
        d = X.shape[1]
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("Covariance matrix must be positive definite")
        diff = X - mean
        sol = np.linalg.solve(cov, diff.T).T
        quad = np.sum(diff * sol, axis=1)
        return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)

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
            log_prob = np.column_stack([
                np.log(self.weights_[j] + 1e-12) + self._log_gauss(X, self.means_[j], self.covariances_[j])
                for j in range(k)
            ])
            log_norm = log_sum_exp(log_prob, axis=1)
            resp = np.exp(log_prob - log_norm[:, None])

            nk = resp.sum(axis=0) + 1e-12
            self.weights_ = nk / n
            self.means_ = (resp.T @ X) / nk[:, None]

            covs = []
            for j in range(k):
                diff = X - self.means_[j]
                cov = (resp[:, j][:, None] * diff).T @ diff / nk[j]
                cov += self.reg_covar * np.eye(d)
                covs.append(cov)
            self.covariances_ = np.asarray(covs)
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
        log_prob = np.column_stack([
            np.log(cast(np.ndarray, self.weights_)[j] + 1e-12) + self._log_gauss(X, cast(np.ndarray, self.means_)[j], cast(np.ndarray, self.covariances_)[j])
            for j in range(self.n_components)
        ])
        log_norm = log_sum_exp(log_prob, axis=1)
        resp = np.exp(log_prob - log_norm[:, None])
        return resp[0] if resp.shape[0] == 1 else resp

    def predict(self, X):
        resp = self.predict_proba(X)
        idx = np.argmax(resp, axis=1)
        return idx[0] if idx.shape[0] == 1 else idx

    def score(self, X, y=None):
        X = check_2d_array(X)
        log_prob = np.column_stack([
            np.log(cast(np.ndarray, self.weights_)[j] + 1e-12) + self._log_gauss(X, cast(np.ndarray, self.means_)[j], cast(np.ndarray, self.covariances_)[j])
            for j in range(self.n_components)
        ])
        return float(np.mean(log_sum_exp(log_prob, axis=1)))
