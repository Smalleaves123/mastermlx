from __future__ import annotations

import numpy as np
from typing import cast

from ..base import BaseEstimator
from ..utils import as_2d, check_2d_array
from .base import VariationalEstimator
from .utils import digamma, has_converged, log_sum_exp, normalize_log_probs


class VariationalGaussianMixture(BaseEstimator, VariationalEstimator):
    """Mean-field variational Gaussian mixture with isotropic precisions."""

    def __init__(
        self,
        n_components=2,
        max_iter=100,
        tol=1e-4,
        alpha0=1.0,
        beta0=1.0,
        a0=1.0,
        b0=1.0,
        random_state=None,
    ):
        VariationalEstimator.__init__(self)
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.random_state = random_state

        self.weights_ = None
        self.means_ = None
        self.resp_ = None
        self.alpha_ = None
        self.beta_ = None
        self.a_ = None
        self.b_ = None

    def _init_params(self, X):
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n_samples, size=self.n_components, replace=False)

        data_mean = np.mean(X, axis=0)
        data_var = float(np.mean(np.var(X, axis=0))) + 1e-6

        self.alpha_ = np.full(self.n_components, self.alpha0, dtype=float)
        self.beta_ = np.full(self.n_components, self.beta0, dtype=float)
        self.a_ = np.full(self.n_components, self.a0 + 0.5 * n_features, dtype=float)
        self.b_ = np.full(self.n_components, self.b0 + 0.5 * data_var, dtype=float)
        self.means_ = X[idx].copy()
        self._m0 = data_mean
        self._n_features = n_features

    def _expected_log_pi(self):
        alpha = cast(np.ndarray, self.alpha_)
        return digamma(alpha) - digamma(np.sum(alpha))

    def _expected_log_lambda(self):
        a = cast(np.ndarray, self.a_)
        b = cast(np.ndarray, self.b_)
        return digamma(a) - np.log(b)

    def _estimate_log_resp(self, X):
        means = cast(np.ndarray, self.means_)
        a = cast(np.ndarray, self.a_)
        b = cast(np.ndarray, self.b_)
        beta = cast(np.ndarray, self.beta_)
        n_samples = X.shape[0]
        log_resp = np.empty((n_samples, self.n_components), dtype=float)
        e_log_pi = self._expected_log_pi()
        e_log_lambda = self._expected_log_lambda()

        for k in range(self.n_components):
            diff = X - means[k]
            sq_norm = np.sum(diff * diff, axis=1)
            expected_quad = (a[k] / b[k]) * sq_norm + (self._n_features / beta[k])
            log_resp[:, k] = (
                e_log_pi[k]
                + 0.5 * self._n_features * e_log_lambda[k]
                - 0.5 * self._n_features * np.log(2.0 * np.pi)
                - 0.5 * expected_quad
            )

        return log_resp

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        n_samples, n_features = X.shape
        if self.n_components < 1 or self.n_components > n_samples:
            raise ValueError("n_components must be between 1 and number of samples")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if self.alpha0 <= 0.0 or self.beta0 <= 0.0 or self.a0 <= 0.0 or self.b0 <= 0.0:
            raise ValueError("prior hyperparameters must be positive")

        self._init_params(X)
        self.lower_bound_ = []
        self.n_iter_ = 0
        self.converged_ = False

        prev = None
        for step_idx in range(1, self.max_iter + 1):
            log_rho = self._estimate_log_resp(X)
            resp, log_norm = normalize_log_probs(log_rho, axis=1)

            nk = resp.sum(axis=0) + 1e-12
            xbar = (resp.T @ X) / nk[:, None]

            sq_scatter = np.zeros(self.n_components, dtype=float)
            for k in range(self.n_components):
                diff = X - xbar[k]
                sq_scatter[k] = np.sum(resp[:, k] * np.sum(diff * diff, axis=1))

            self.alpha_ = self.alpha0 + nk
            self.beta_ = self.beta0 + nk
            self.means_ = (self.beta0 * self._m0[None, :] + nk[:, None] * xbar) / self.beta_[:, None]
            self.a_ = self.a0 + 0.5 * n_features * nk

            prior_shift = np.sum((xbar - self._m0[None, :]) ** 2, axis=1)
            self.b_ = self.b0 + 0.5 * (sq_scatter + (self.beta0 * nk / self.beta_) * prior_shift)

            self.resp_ = resp
            self.weights_ = self.alpha_ / np.sum(self.alpha_)

            lb = float(np.mean(log_norm))
            self.lower_bound_.append(lb)
            self.n_iter_ = step_idx
            self.converged_ = has_converged(lb, prev, self.tol)
            if self.converged_:
                break
            prev = lb

        return self

    def predict_proba(self, X):
        if self.means_ is None or self.weights_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X).astype(float)
        log_rho = self._estimate_log_resp(X)
        resp, _ = normalize_log_probs(log_rho, axis=1)
        return resp[0] if resp.shape[0] == 1 else resp

    def predict(self, X):
        resp = self.predict_proba(X)
        if resp.ndim == 1:
            return int(np.argmax(resp))
        return np.argmax(resp, axis=1)

    def _posterior_summary(self):
        means = cast(np.ndarray, self.means_)
        weights = cast(np.ndarray, self.weights_)
        return {
            "n_components": int(self.n_components),
            "mean_dim": int(means.shape[1]),
            "min_weight": float(np.min(weights)),
            "max_weight": float(np.max(weights)),
        }

    def score(self, X, y=None):
        X = check_2d_array(X).astype(float)
        log_rho = self._estimate_log_resp(X)
        return float(np.mean(log_sum_exp(log_rho, axis=1)))


class BayesianGaussianMixture(VariationalGaussianMixture):
    """Bayesian Gaussian mixture with variational inference."""

    def __init__(
        self,
        n_components=2,
        max_iter=100,
        tol=1e-4,
        weight_concentration_prior=1.0,
        mean_precision_prior=1.0,
        precision_shape_prior=1.0,
        precision_rate_prior=1.0,
        random_state=None,
    ):
        super().__init__(
            n_components=n_components,
            max_iter=max_iter,
            tol=tol,
            alpha0=weight_concentration_prior,
            beta0=mean_precision_prior,
            a0=precision_shape_prior,
            b0=precision_rate_prior,
            random_state=random_state,
        )
        self.weight_concentration_prior = float(weight_concentration_prior)
        self.mean_precision_prior = float(mean_precision_prior)
        self.precision_shape_prior = float(precision_shape_prior)
        self.precision_rate_prior = float(precision_rate_prior)
        self.active_components_ = None
        self.n_active_components_ = None

    def fit(self, X, y=None):
        super().fit(X, y=y)
        threshold = 1.0 / max(10.0 * self.n_components, 100.0)
        weights = cast(np.ndarray, self.weights_)
        self.active_components_ = weights > threshold
        self.n_active_components_ = int(np.sum(cast(np.ndarray, self.active_components_)))
        return self

    def _posterior_summary(self):
        summary = super()._posterior_summary()
        summary.update({
            "n_active_components": int(cast(int, self.n_active_components_)),
            "weight_concentration_prior": float(self.weight_concentration_prior),
        })
        return summary


VGMM = VariationalGaussianMixture
BayesGMM = BayesianGaussianMixture
