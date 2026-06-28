from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import as_2d, check_1d_array, check_2d_array, check_same_rows, r2_score
from ._kernels import pairwise_kernel, resolve_gamma


class KernelSVR(BaseEstimator):
    """Kernel support vector regressor with epsilon-insensitive loss."""

    def __init__(
        self,
        C=1.0,
        epsilon=0.1,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=0.0,
        lr=0.05,
        max_iter=2000,
        tol=1e-6,
    ):
        self.C = float(C)
        self.epsilon = float(epsilon)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.X_ = None
        self.y_ = None
        self.alpha_ = None
        self.alpha_star_ = None
        self.dual_coef_ = None
        self.intercept_ = 0.0
        self.loss_ = []

    def _kernel(self, X, Y):
        return pairwise_kernel(X, Y, kernel=self.kernel, gamma=self._gamma, coef0=self.coef0, degree=self.degree)

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y).astype(float)
        X, y = check_same_rows(X, y)
        if self.C <= 0:
            raise ValueError("C must be positive")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")

        n_samples, n_features = X.shape
        self._gamma = resolve_gamma(self.gamma, n_features)
        self.X_ = X
        self.y_ = y
        self.alpha_ = np.zeros(n_samples, dtype=float)
        self.alpha_star_ = np.zeros(n_samples, dtype=float)
        self.loss_ = []

        K = self._kernel(X, X)
        prev = None

        for step_idx in range(self.max_iter):
            diff = self.alpha_ - self.alpha_star_
            grad = K @ diff + self.epsilon - y
            grad_star = -K @ diff + self.epsilon + y
            step = self.lr / np.sqrt(step_idx + 1.0)

            self.alpha_ = np.clip(self.alpha_ - step * grad, 0.0, self.C)
            self.alpha_star_ = np.clip(self.alpha_star_ - step * grad_star, 0.0, self.C)

            diff = self.alpha_ - self.alpha_star_
            shift = np.mean(diff)
            self.alpha_ = np.clip(self.alpha_ - np.maximum(shift, 0.0), 0.0, self.C)
            self.alpha_star_ = np.clip(self.alpha_star_ + np.minimum(shift, 0.0), 0.0, self.C)

            diff = self.alpha_ - self.alpha_star_
            bias = np.mean(diff)
            if abs(bias) > 1e-10:
                move = min(abs(bias) / 2.0, self.C)
                if bias > 0:
                    self.alpha_ = np.clip(self.alpha_ - move, 0.0, self.C)
                    self.alpha_star_ = np.clip(self.alpha_star_ + move, 0.0, self.C)
                else:
                    self.alpha_ = np.clip(self.alpha_ + move, 0.0, self.C)
                    self.alpha_star_ = np.clip(self.alpha_star_ - move, 0.0, self.C)

            diff = self.alpha_ - self.alpha_star_
            pred = K @ diff
            err = pred - y
            loss = 0.5 * diff @ pred + self.epsilon * np.sum(self.alpha_ + self.alpha_star_) - y @ diff
            self.loss_.append(float(loss))
            if prev is not None and abs(prev - loss) < self.tol:
                break
            prev = loss

        self.dual_coef_ = self.alpha_ - self.alpha_star_
        pred = K @ self.dual_coef_
        margin = np.abs(np.abs(pred - y) - self.epsilon) <= max(self.epsilon * 0.5, 1e-3)
        free = ((self.alpha_ > 1e-6) & (self.alpha_ < self.C - 1e-6)) | (
            (self.alpha_star_ > 1e-6) & (self.alpha_star_ < self.C - 1e-6)
        )
        mask = margin | free
        if np.any(mask):
            self.intercept_ = float(np.mean(y[mask] - pred[mask]))
        else:
            self.intercept_ = float(np.mean(y - pred))
        return self

    def predict(self, X):
        if self.dual_coef_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X).astype(float)
        K = self._kernel(X, self.X_)
        pred = K @ self.dual_coef_ + self.intercept_
        return float(pred[0]) if pred.shape[0] == 1 else pred

    def score(self, X, y):
        return r2_score(y, self.predict(X))
