from __future__ import annotations

import math
from typing import cast

import numpy as np

from ..base import BaseEstimator
from ..utils import as_2d, check_1d_array, check_2d_array, check_same_rows, r2_score
from .base import VariationalEstimator
from .utils import digamma, has_converged, log_gamma


class VariationalLinearRegression(BaseEstimator, VariationalEstimator):
    """Variational Bayesian linear regression with Gamma-Gaussian priors."""

    def __init__(
        self,
        a0=1.0,
        b0=1.0,
        c0=1.0,
        d0=1.0,
        fit_intercept=True,
        max_iter=200,
        tol=1e-5,
    ):
        VariationalEstimator.__init__(self)
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.c0 = float(c0)
        self.d0 = float(d0)
        self.fit_intercept = fit_intercept
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.coef_ = None
        self.intercept_ = None
        self.posterior_mean_ = None
        self.posterior_cov_ = None
        self.noise_shape_ = None
        self.noise_rate_ = None
        self.weight_shape_ = None
        self.weight_rate_ = None

    def _add_bias(self, X):
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X

    def _expected_noise_precision(self):
        return cast(float, self.noise_shape_) / cast(float, self.noise_rate_)

    def _expected_weight_precision(self):
        return cast(float, self.weight_shape_) / cast(float, self.weight_rate_)

    def _elbo(self, X, y):
        posterior_mean = cast(np.ndarray, self.posterior_mean_)
        posterior_cov = cast(np.ndarray, self.posterior_cov_)
        noise_shape = cast(float, self.noise_shape_)
        noise_rate = cast(float, self.noise_rate_)
        weight_shape = cast(float, self.weight_shape_)
        weight_rate = cast(float, self.weight_rate_)
        n_samples, n_features = X.shape
        e_alpha = self._expected_noise_precision()
        e_lambda = self._expected_weight_precision()
        e_log_alpha = float(digamma(noise_shape) - np.log(noise_rate))
        e_log_lambda = float(digamma(weight_shape) - np.log(weight_rate))

        xtx = X.T @ X
        residual = y - X @ posterior_mean
        sq_error = residual @ residual + np.trace(xtx @ posterior_cov)
        weight_sq = posterior_mean @ posterior_mean + np.trace(posterior_cov)

        e_log_p_y = 0.5 * n_samples * (e_log_alpha - np.log(2.0 * np.pi)) - 0.5 * e_alpha * sq_error
        e_log_p_w = 0.5 * n_features * (e_log_lambda - np.log(2.0 * np.pi)) - 0.5 * e_lambda * weight_sq
        e_log_p_alpha = (
            self.a0 * np.log(self.b0)
            - math.lgamma(self.a0)
            + (self.a0 - 1.0) * e_log_alpha
            - self.b0 * e_alpha
        )
        e_log_p_lambda = (
            self.c0 * np.log(self.d0)
            - math.lgamma(self.c0)
            + (self.c0 - 1.0) * e_log_lambda
            - self.d0 * e_lambda
        )

        sign, logdet = np.linalg.slogdet(posterior_cov)
        if sign <= 0:
            raise ValueError("posterior covariance must be positive definite")
        entropy_w = 0.5 * (n_features * (1.0 + np.log(2.0 * np.pi)) + logdet)
        entropy_alpha = (
            noise_shape
            - np.log(noise_rate)
            + log_gamma(np.array(noise_shape))
            + (1.0 - noise_shape) * digamma(noise_shape)
        )
        entropy_lambda = (
            weight_shape
            - np.log(weight_rate)
            + log_gamma(np.array(weight_shape))
            + (1.0 - weight_shape) * digamma(weight_shape)
        )
        return float(e_log_p_y + e_log_p_w + e_log_p_alpha + e_log_p_lambda + entropy_w + entropy_alpha + entropy_lambda)

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y).astype(float)
        X, y = check_same_rows(X, y)
        if min(self.a0, self.b0, self.c0, self.d0) <= 0.0:
            raise ValueError("prior hyperparameters must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if self.tol <= 0.0:
            raise ValueError("tol must be positive")

        Xb = self._add_bias(X)
        n_samples, n_features = Xb.shape
        xtx = Xb.T @ Xb

        self.posterior_mean_ = np.zeros(n_features, dtype=float)
        self.posterior_cov_ = np.eye(n_features, dtype=float)
        self.noise_shape_ = self.a0 + 0.5 * n_samples
        self.noise_rate_ = self.b0 + 0.5 * np.var(y)
        self.weight_shape_ = self.c0 + 0.5 * n_features
        self.weight_rate_ = self.d0 + 0.5
        self.lower_bound_ = []
        self.n_iter_ = 0
        self.converged_ = False

        prev = None
        for step_idx in range(1, self.max_iter + 1):
            e_alpha = self._expected_noise_precision()
            e_lambda = self._expected_weight_precision()

            precision = e_alpha * xtx + e_lambda * np.eye(n_features)
            self.posterior_cov_ = np.linalg.pinv(precision)
            self.posterior_mean_ = e_alpha * (self.posterior_cov_ @ Xb.T @ y)

            residual = y - Xb @ self.posterior_mean_
            sq_error = residual @ residual + np.trace(xtx @ self.posterior_cov_)
            weight_sq = self.posterior_mean_ @ self.posterior_mean_ + np.trace(self.posterior_cov_)

            self.noise_rate_ = self.b0 + 0.5 * sq_error
            self.weight_rate_ = self.d0 + 0.5 * weight_sq

            lb = self._elbo(Xb, y)
            self.lower_bound_.append(lb)
            self.n_iter_ = step_idx
            self.converged_ = has_converged(lb, prev, self.tol)
            if self.converged_:
                break
            prev = lb

        if self.fit_intercept:
            self.intercept_ = float(self.posterior_mean_[0])
            self.coef_ = self.posterior_mean_[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self.posterior_mean_
        return self

    def predict(self, X, return_std=False):
        X = as_2d(X).astype(float)
        if self.posterior_mean_ is None or self.posterior_cov_ is None:
            raise RuntimeError("Model has not been fit yet")

        posterior_mean = cast(np.ndarray, self.posterior_mean_)
        posterior_cov = cast(np.ndarray, self.posterior_cov_)
        Xb = self._add_bias(X)
        mean = Xb @ posterior_mean
        if not return_std:
            return float(mean[0]) if mean.shape[0] == 1 else mean

        noise_var = 1.0 / self._expected_noise_precision()
        var = noise_var + np.sum((Xb @ posterior_cov) * Xb, axis=1)
        std = np.sqrt(np.maximum(var, 0.0))
        if mean.shape[0] == 1:
            return float(mean[0]), float(std[0])
        return mean, std

    def _posterior_summary(self):
        posterior_mean = cast(np.ndarray, self.posterior_mean_)
        return {
            "noise_shape": float(cast(float, self.noise_shape_)),
            "noise_rate": float(cast(float, self.noise_rate_)),
            "weight_shape": float(cast(float, self.weight_shape_)),
            "weight_rate": float(cast(float, self.weight_rate_)),
            "posterior_dim": int(posterior_mean.shape[0]),
        }

    def sample_posterior_predictive(self, X, n_samples=1, random_state=None):
        X = as_2d(X).astype(float)
        weights = self.sample_posterior_weights(n_samples=n_samples, random_state=random_state)
        Xb = self._add_bias(X)
        noise_std = np.sqrt(1.0 / self._expected_noise_precision())
        rng = np.random.default_rng(random_state)

        if int(n_samples) == 1:
            mean = Xb @ weights
            sample = mean + rng.normal(scale=noise_std, size=Xb.shape[0])
            return float(sample[0]) if sample.shape[0] == 1 else sample

        mean = weights @ Xb.T
        sample = mean + rng.normal(scale=noise_std, size=mean.shape)
        return sample[:, 0] if Xb.shape[0] == 1 else sample

    def score(self, X, y):
        return r2_score(y, self.predict(X))


VLinReg = VariationalLinearRegression
