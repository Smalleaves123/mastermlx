from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import accuracy, as_2d, check_1d_array, check_2d_array, check_same_rows
from ._kernels import pairwise_kernel, resolve_gamma


class _BinaryNuSVC:
    def __init__(self, nu=0.5, kernel="rbf", degree=3, gamma=None, coef0=0.0,
                 tol=1e-3, max_iter=1000, random_state=None):
        self.nu = float(nu)
        self.kernel = kernel
        self.degree = int(degree)
        self.gamma = gamma
        self.coef0 = float(coef0)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.alphas_ = None
        self.b_ = 0.0
        self.support_ = None

    def fit(self, X, y):
        n = X.shape[0]
        self._gamma = resolve_gamma(self.gamma, X.shape[1])
        K = pairwise_kernel(X, X, self.kernel, self._gamma, self.coef0, self.degree)
        y = np.where(y > 0, 1.0, -1.0)

        # ν-SVM dual: 0 ≤ α_i ≤ 1/n, Σ α_i y_i = 0, Σ α_i ≥ ν
        rng = np.random.default_rng(self.random_state)
        alpha = np.zeros(n)
        self.b_ = 0.0

        for _ in range(self.max_iter):
            changed = 0
            for i in range(n):
                Ei = np.sum(alpha * y * K[:, i]) + self.b_ - y[i]
                if (y[i] * Ei < -self.tol and alpha[i] < 1.0 / n) or \
                   (y[i] * Ei > self.tol and alpha[i] > 0):
                    j = int(rng.integers(0, n - 1))
                    if j >= i:
                        j += 1
                    Ej = np.sum(alpha * y * K[:, j]) + self.b_ - y[j]
                    ai_old, aj_old = alpha[i], alpha[j]

                    if y[i] != y[j]:
                        L = max(0.0, aj_old - ai_old)
                        H = min(1.0 / n, 1.0 / n + aj_old - ai_old)
                    else:
                        L = max(0.0, ai_old + aj_old - 1.0 / n)
                        H = min(1.0 / n, ai_old + aj_old)
                    if L >= H:
                        continue

                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    aj = np.clip(aj_old - y[j] * (Ei - Ej) / eta, L, H)
                    if abs(aj - aj_old) < 1e-8:
                        continue
                    alpha[i] = ai_old + y[i] * y[j] * (aj_old - aj)
                    alpha[j] = aj

                    b1 = self.b_ - Ei - y[i]*(alpha[i]-ai_old)*K[i,i] - y[j]*(alpha[j]-aj_old)*K[i,j]
                    b2 = self.b_ - Ej - y[i]*(alpha[i]-ai_old)*K[i,j] - y[j]*(alpha[j]-aj_old)*K[j,j]
                    if 0 < alpha[i] < 1.0/n: self.b_ = b1
                    elif 0 < alpha[j] < 1.0/n: self.b_ = b2
                    else: self.b_ = (b1 + b2) / 2.0
                    changed += 1
            if changed == 0:
                break

        self.alphas_ = alpha
        self.support_ = alpha > 1e-8
        self.X_ = X
        self.y_ = y
        return self

    def decision_function(self, X):
        K = pairwise_kernel(X, self.X_[self.support_], self.kernel, self._gamma, self.coef0, self.degree)
        return K @ (self.alphas_[self.support_] * self.y_[self.support_]) + self.b_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


class NuSVC(BaseEstimator):
    """ν-Support Vector Classification."""

    def __init__(self, nu=0.5, kernel="rbf", degree=3, gamma=None, coef0=0.0,
                 tol=1e-3, max_iter=1000, random_state=None):
        self.nu = nu
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.classes_ = None

    def fit(self, X, y=None):
        X, y = check_same_rows(check_2d_array(X), check_1d_array(y))
        self.classes_ = np.unique(y)
        if self.classes_.size == 2:
            y_bin = np.where(y == self.classes_[1], 1.0, -1.0)
            self._binary = _BinaryNuSVC(nu=self.nu, kernel=self.kernel, degree=self.degree,
                                         gamma=self.gamma, coef0=self.coef0, tol=self.tol,
                                         max_iter=self.max_iter, random_state=self.random_state)
            self._binary.fit(X, y_bin)
        else:
            self._binaries = []
            for c in self.classes_:
                y_bin = np.where(y == c, 1.0, -1.0)
                b = _BinaryNuSVC(nu=self.nu, kernel=self.kernel, degree=self.degree,
                                  gamma=self.gamma, coef0=self.coef0, tol=self.tol,
                                  max_iter=self.max_iter, random_state=self.random_state)
                b.fit(X, y_bin)
                self._binaries.append(b)
        return self

    def predict(self, X):
        if self.classes_.size == 2:
            p = self._binary.predict(X)
            return np.where(p > 0, self.classes_[1], self.classes_[0])
        scores = np.column_stack([b.decision_function(X) for b in self._binaries])
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X, y):
        return accuracy(y, self.predict(X))
