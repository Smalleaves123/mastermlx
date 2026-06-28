from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import accuracy, as_2d, check_1d_array, check_2d_array, check_same_rows
from ..utils.math import sigmoid
from .base import VariationalEstimator
from .utils import has_converged


def _lambda_xi(xi):
    xi = np.asarray(xi, dtype=float)
    safe_xi = np.maximum(xi, 1e-12)
    return np.tanh(safe_xi / 2.0) / (4.0 * safe_xi)


class VariationalLogisticRegression(BaseEstimator, VariationalEstimator):
    """Variational Bayesian logistic regression for binary classification."""

    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=200, tol=1e-5):
        VariationalEstimator.__init__(self)
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None
        self.posterior_mean_ = None
        self.posterior_cov_ = None
        self.xi_ = None

    def _add_bias(self, X):
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X

    def _expected_linear_square(self, Xb):
        second_moment = self.posterior_cov_ + np.outer(self.posterior_mean_, self.posterior_mean_)
        return np.sum((Xb @ second_moment) * Xb, axis=1)

    def _elbo(self, Xb, y_bin):
        linear_mean = Xb @ self.posterior_mean_
        expected_square = self._expected_linear_square(Xb)
        lam = _lambda_xi(self.xi_)
        log_sigmoid_xi = np.log(sigmoid(self.xi_))

        log_lik_bound = np.sum(
            log_sigmoid_xi
            + (y_bin - 0.5) * linear_mean
            - 0.5 * self.xi_
            - lam * (expected_square - self.xi_**2)
        )

        dim = self.posterior_mean_.shape[0]
        weight_sq = self.posterior_mean_ @ self.posterior_mean_ + np.trace(self.posterior_cov_)
        log_prior = 0.5 * dim * (np.log(self.alpha) - np.log(2.0 * np.pi)) - 0.5 * self.alpha * weight_sq

        sign, logdet = np.linalg.slogdet(self.posterior_cov_)
        if sign <= 0:
            raise ValueError("posterior covariance must be positive definite")
        entropy = 0.5 * (dim * (1.0 + np.log(2.0 * np.pi)) + logdet)
        return float(log_lik_bound + log_prior + entropy)

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y)
        X, y = check_same_rows(X, y)
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if self.tol <= 0.0:
            raise ValueError("tol must be positive")

        classes = np.unique(y)
        if classes.shape[0] != 2:
            raise ValueError("VariationalLogisticRegression currently supports binary targets only")
        self.classes_ = classes
        y_bin = (y == classes[1]).astype(float)

        Xb = self._add_bias(X)
        n_samples, n_features = Xb.shape
        prior_precision = self.alpha * np.eye(n_features)
        target_shift = y_bin - 0.5

        self.posterior_mean_ = np.zeros(n_features, dtype=float)
        self.posterior_cov_ = np.eye(n_features, dtype=float) / self.alpha
        self.xi_ = np.ones(n_samples, dtype=float)
        self.lower_bound_ = []
        self.n_iter_ = 0
        self.converged_ = False

        prev = None
        for step_idx in range(1, self.max_iter + 1):
            lam = _lambda_xi(self.xi_)
            weighted_xtx = Xb.T @ (2.0 * lam[:, None] * Xb)
            precision = prior_precision + weighted_xtx
            self.posterior_cov_ = np.linalg.pinv(precision)
            self.posterior_mean_ = self.posterior_cov_ @ Xb.T @ target_shift

            self.xi_ = np.sqrt(np.maximum(self._expected_linear_square(Xb), 1e-12))

            lb = self._elbo(Xb, y_bin)
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

    def decision_function(self, X):
        X = as_2d(X).astype(float)
        if self.posterior_mean_ is None:
            raise RuntimeError("Model has not been fit yet")
        Xb = self._add_bias(X)
        scores = Xb @ self.posterior_mean_
        return float(scores[0]) if scores.shape[0] == 1 else scores

    def predict_proba(self, X):
        scores = self.decision_function(X)
        scores = np.asarray(scores, dtype=float)
        if scores.ndim == 0:
            p1 = float(sigmoid(scores))
            return np.array([[1.0 - p1, p1]])
        p1 = sigmoid(scores)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def sample_posterior_predict_proba(self, X, n_samples=1, random_state=None):
        X = as_2d(X).astype(float)
        weights = self.sample_posterior_weights(n_samples=n_samples, random_state=random_state)
        Xb = self._add_bias(X)

        if int(n_samples) == 1:
            scores = Xb @ weights
            p1 = sigmoid(scores)
            probs = np.column_stack([1.0 - p1, p1])
            return probs[0] if probs.shape[0] == 1 else probs

        scores = weights @ Xb.T
        p1 = sigmoid(scores)
        probs = np.stack([1.0 - p1, p1], axis=-1)
        return probs[:, 0, :] if Xb.shape[0] == 1 else probs

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = (proba[:, 1] >= 0.5).astype(int)
        pred = self.classes_[idx]
        return pred[0] if pred.shape[0] == 1 else pred

    def _posterior_summary(self):
        return {
            "alpha": float(self.alpha),
            "posterior_dim": int(self.posterior_mean_.shape[0]),
            "mean_norm": float(np.linalg.norm(self.posterior_mean_)),
        }

    def score(self, X, y):
        return accuracy(y, self.predict(X))


VLogReg = VariationalLogisticRegression
