from __future__ import annotations

import math

import numpy as np

from ..base import BaseEstimator
from ..utils import as_2d, check_1d_array, check_2d_array, check_same_rows, r2_score
from .base import VariationalEstimator
from .utils import has_converged


class VariationalPoissonRegression(BaseEstimator, VariationalEstimator):
    """Mean-field Gaussian variational Poisson regression with log link."""

    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=300, tol=1e-5, lr=0.05):
        VariationalEstimator.__init__(self)
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.lr = float(lr)

        self.coef_ = None
        self.intercept_ = None
        self.posterior_mean_ = None
        self.posterior_cov_ = None
        self.posterior_var_ = None

    def _add_bias(self, X):
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X

    def _expected_stats_from_params(self, Xb, mean, var):
        eta_mean = Xb @ mean
        eta_var = np.sum((Xb**2) * var[None, :], axis=1)
        exp_term = np.exp(np.clip(eta_mean + 0.5 * eta_var, -50.0, 50.0))
        return eta_mean, eta_var, exp_term

    def _expected_stats(self, Xb):
        return self._expected_stats_from_params(Xb, self.posterior_mean_, self.posterior_var_)

    def _elbo_from_params(self, Xb, y, mean, var):
        eta_mean, _, exp_term = self._expected_stats_from_params(Xb, mean, var)
        data_term = np.sum(y * eta_mean - exp_term - np.array([math.lgamma(float(v) + 1.0) for v in y]))
        dim = mean.shape[0]
        weight_sq = np.sum(mean**2 + var)
        log_prior = 0.5 * dim * (np.log(self.alpha) - np.log(2.0 * np.pi)) - 0.5 * self.alpha * weight_sq
        entropy = 0.5 * np.sum(np.log(2.0 * np.pi * np.e * var))
        return float(data_term + log_prior + entropy)

    def _elbo(self, Xb, y):
        return self._elbo_from_params(Xb, y, self.posterior_mean_, self.posterior_var_)

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y).astype(float)
        X, y = check_same_rows(X, y)
        if np.any(y < 0.0):
            raise ValueError("y must be non-negative")
        if np.any(np.abs(y - np.round(y)) > 1e-8):
            raise ValueError("y must contain count-like values")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if self.tol <= 0.0:
            raise ValueError("tol must be positive")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")

        Xb = self._add_bias(X)
        _, n_features = Xb.shape

        target = np.log(y + 1.0)
        init_mean, *_ = np.linalg.lstsq(Xb, target, rcond=None)
        self.posterior_mean_ = init_mean.astype(float)
        self.posterior_var_ = np.full(n_features, 1.0 / self.alpha, dtype=float)
        self.posterior_cov_ = np.diag(self.posterior_var_)
        self.lower_bound_ = []
        self.n_iter_ = 0
        self.converged_ = False

        prev = None
        for step_idx in range(1, self.max_iter + 1):
            _, _, exp_term = self._expected_stats(Xb)

            grad_mean = Xb.T @ (y - exp_term) - self.alpha * self.posterior_mean_
            grad_var = -0.5 * (Xb**2).T @ exp_term - 0.5 * self.alpha + 0.5 / self.posterior_var_
            current_elbo = self._elbo(Xb, y)

            step = self.lr
            updated = False
            clipped_mean_grad = np.clip(grad_mean, -100.0, 100.0)
            clipped_var_grad = np.clip(grad_var * self.posterior_var_, -5.0, 5.0)
            current_log_var = np.log(np.maximum(self.posterior_var_, 1e-12))
            current_mean = self.posterior_mean_.copy()
            current_var = self.posterior_var_.copy()

            for _ in range(20):
                cand_mean = current_mean + step * clipped_mean_grad
                cand_log_var = current_log_var + step * clipped_var_grad
                cand_var = np.maximum(np.exp(cand_log_var), 1e-8)
                cand_elbo = self._elbo_from_params(Xb, y, cand_mean, cand_var)
                if cand_elbo >= current_elbo - 1e-10:
                    self.posterior_mean_ = cand_mean
                    self.posterior_var_ = cand_var
                    updated = True
                    lb = cand_elbo
                    break
                step *= 0.5

            if not updated:
                lb = current_elbo

            self.posterior_cov_ = np.diag(self.posterior_var_)
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

        Xb = self._add_bias(X)
        eta_mean = Xb @ self.posterior_mean_
        eta_var = np.sum((Xb**2) * self.posterior_var_[None, :], axis=1)
        mean = np.exp(np.clip(eta_mean + 0.5 * eta_var, -50.0, 50.0))
        if not return_std:
            return float(mean[0]) if mean.shape[0] == 1 else mean

        variance = mean + mean**2 * (np.exp(np.clip(eta_var, 0.0, 50.0)) - 1.0)
        std = np.sqrt(np.maximum(variance, 0.0))
        if mean.shape[0] == 1:
            return float(mean[0]), float(std[0])
        return mean, std

    def _posterior_summary(self):
        return {
            "alpha": float(self.alpha),
            "posterior_dim": int(self.posterior_mean_.shape[0]),
            "avg_posterior_var": float(np.mean(self.posterior_var_)),
        }

    def score(self, X, y):
        return r2_score(y, self.predict(X))


VPoisReg = VariationalPoissonRegression
