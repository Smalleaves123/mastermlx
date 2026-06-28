from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import as_2d, check_2d_array
from ._kernels import pairwise_kernel, resolve_gamma


def _project_capped_simplex(x, cap, target=1.0, tol=1e-9, max_iter=100):
    low = np.min(x) - cap
    high = np.max(x)
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        clipped = np.clip(x - mid, 0.0, cap)
        total = np.sum(clipped)
        if abs(total - target) < tol:
            return clipped
        if total > target:
            low = mid
        else:
            high = mid
    return np.clip(x - 0.5 * (low + high), 0.0, cap)


class OneClassSVM(BaseEstimator):
    """Kernel one-class SVM for outlier detection."""

    def __init__(self, nu=0.1, kernel="rbf", gamma=None, degree=3, coef0=0.0, lr=0.1, max_iter=2000, tol=1e-6):
        self.nu = float(nu)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.X_ = None
        self.alpha_ = None
        self.intercept_ = 0.0
        self.support_indices_ = None
        self.support_vectors_ = None
        self.score_samples_ = None

    def _kernel(self, X, Y):
        return pairwise_kernel(X, Y, kernel=self.kernel, gamma=self._gamma, coef0=self.coef0, degree=self.degree)

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        n_samples, n_features = X.shape
        if not 0.0 < self.nu <= 1.0:
            raise ValueError("nu must be in (0, 1]")

        self._gamma = resolve_gamma(self.gamma, n_features)
        self.X_ = X
        K = self._kernel(X, X)

        cap = 1.0 / (self.nu * n_samples)
        alpha = np.full(n_samples, 1.0 / n_samples, dtype=float)
        prev = None

        for step_idx in range(self.max_iter):
            grad = K @ alpha
            step = self.lr / np.sqrt(step_idx + 1.0)
            alpha = _project_capped_simplex(alpha - step * grad, cap)
            obj = 0.5 * alpha @ (K @ alpha)
            if prev is not None and abs(prev - obj) < self.tol:
                break
            prev = obj

        self.alpha_ = alpha
        scores = K @ self.alpha_
        free = (self.alpha_ > 1e-6) & (self.alpha_ < cap - 1e-6)
        if np.any(free):
            self.intercept_ = float(np.mean(scores[free]))
        else:
            self.intercept_ = float(np.quantile(scores, self.nu))

        self.support_indices_ = np.flatnonzero(self.alpha_ > 1e-6)
        self.support_vectors_ = self.X_[self.support_indices_]
        self.score_samples_ = scores - self.intercept_
        return self

    def decision_function(self, X):
        if self.alpha_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X).astype(float)
        K = self._kernel(X, self.X_)
        scores = K @ self.alpha_ - self.intercept_
        return float(scores[0]) if scores.shape[0] == 1 else scores

    def predict(self, X):
        scores = self.decision_function(X)
        pred = np.where(scores >= 0.0, 1, -1)
        return int(pred) if np.ndim(pred) == 0 else pred

    def score_samples(self, X):
        return self.decision_function(X)
