from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import as_2d, check_2d_array


class EllipticEnvelope(BaseEstimator):
    """Robust covariance envelope using iterative support trimming."""

    def __init__(self, contamination=0.1, support_fraction=None, max_iter=25, reg_covar=1e-6):
        self.contamination = float(contamination)
        self.support_fraction = support_fraction
        self.max_iter = int(max_iter)
        self.reg_covar = float(reg_covar)
        self.location_ = None
        self.covariance_ = None
        self.precision_ = None
        self.support_ = None
        self.offset_ = None

    def _fit_covariance(self, X):
        mean = np.mean(X, axis=0)
        Xc = X - mean
        cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
        cov = cov + self.reg_covar * np.eye(X.shape[1])
        return mean, cov

    def _mahalanobis(self, X, mean, precision):
        diff = X - mean
        return np.sum((diff @ precision) * diff, axis=1)

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        n_samples, n_features = X.shape
        if not 0.0 < self.contamination < 0.5:
            raise ValueError("contamination must be in (0, 0.5)")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if self.reg_covar <= 0.0:
            raise ValueError("reg_covar must be positive")

        if self.support_fraction is None:
            frac = max(0.5, 1.0 - self.contamination)
        else:
            frac = float(self.support_fraction)
            if not 0.5 <= frac <= 1.0:
                raise ValueError("support_fraction must be in [0.5, 1.0]")
        keep = max(n_features + 1, int(np.ceil(frac * n_samples)))
        keep = min(keep, n_samples)

        support = np.ones(n_samples, dtype=bool)
        prev = None

        for _ in range(self.max_iter):
            mean, cov = self._fit_covariance(X[support])
            precision = np.linalg.pinv(cov)
            dist = self._mahalanobis(X, mean, precision)
            order = np.argsort(dist)
            new_support = np.zeros(n_samples, dtype=bool)
            new_support[order[:keep]] = True
            if prev is not None and np.array_equal(new_support, prev):
                support = new_support
                break
            prev = support
            support = new_support

        self.location_, self.covariance_ = self._fit_covariance(X[support])
        self.precision_ = np.linalg.pinv(self.covariance_)
        self.support_ = support
        scores = self.score_samples(X)
        self.offset_ = float(np.quantile(scores, 1.0 - self.contamination))
        return self

    def score_samples(self, X):
        if self.location_ is None or self.precision_ is None:
            raise RuntimeError("EllipticEnvelope has not been fit yet")
        X = as_2d(X).astype(float)
        if X.shape[1] != self.location_.shape[0]:
            raise ValueError("X has a different number of features than the fitted data")
        scores = self._mahalanobis(X, self.location_, self.precision_)
        return float(scores[0]) if scores.shape[0] == 1 else scores

    def decision_function(self, X):
        scores = self.score_samples(X)
        out = self.offset_ - scores
        return float(out) if np.ndim(out) == 0 else out

    def predict(self, X):
        scores = self.score_samples(X)
        pred = np.where(scores >= self.offset_, -1, 1)
        return int(pred) if np.ndim(pred) == 0 else pred
