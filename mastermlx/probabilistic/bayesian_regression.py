from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import as_2d, check_1d_array, check_2d_array, check_same_rows, r2_score


class BayesianLinearRegression(BaseEstimator):
    """Bayesian linear regression with isotropic Gaussian prior."""

    def __init__(self, alpha=1.0, beta=1.0, fit_intercept=True):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.posterior_mean_ = None
        self.posterior_cov_ = None

    def _add_bias(self, X):
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y).astype(float)
        X, y = check_same_rows(X, y)
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive")

        Xb = self._add_bias(X)
        n_features = Xb.shape[1]
        precision = self.alpha * np.eye(n_features) + self.beta * (Xb.T @ Xb)
        cov = np.linalg.pinv(precision)
        mean = self.beta * cov @ Xb.T @ y

        self.posterior_cov_ = cov
        self.posterior_mean_ = mean
        if self.fit_intercept:
            self.intercept_ = float(mean[0])
            self.coef_ = mean[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = mean
        return self

    def predict(self, X, return_std=False):
        X = as_2d(X).astype(float)
        if self.posterior_mean_ is None:
            raise RuntimeError("Model has not been fit yet")
        Xb = self._add_bias(X)
        mean = Xb @ self.posterior_mean_
        if not return_std:
            return float(mean[0]) if mean.shape[0] == 1 else mean

        var = 1.0 / self.beta + np.sum((Xb @ self.posterior_cov_) * Xb, axis=1)
        std = np.sqrt(np.maximum(var, 0.0))
        if mean.shape[0] == 1:
            return float(mean[0]), float(std[0])
        return mean, std

    def posterior_summary(self):
        if self.posterior_mean_ is None or self.posterior_cov_ is None:
            raise RuntimeError("Model has not been fit yet")
        return {
            "model": self.__class__.__name__,
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "posterior_dim": int(self.posterior_mean_.shape[0]),
            "fit_intercept": bool(self.fit_intercept),
        }

    def sample_posterior_weights(self, n_samples=1, random_state=None):
        if self.posterior_mean_ is None or self.posterior_cov_ is None:
            raise RuntimeError("Model has not been fit yet")
        rng = np.random.default_rng(random_state)
        samples = rng.multivariate_normal(self.posterior_mean_, self.posterior_cov_, size=int(n_samples))
        return samples[0] if int(n_samples) == 1 else samples

    def sample_posterior_predictive(self, X, n_samples=1, random_state=None):
        X = as_2d(X).astype(float)
        weights = self.sample_posterior_weights(n_samples=n_samples, random_state=random_state)
        Xb = self._add_bias(X)
        noise_std = np.sqrt(1.0 / self.beta)
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


BayesLinReg = BayesianLinearRegression
