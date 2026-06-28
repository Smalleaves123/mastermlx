from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import accuracy, as_2d, check_1d_array, check_2d_array, check_same_rows
from ._kernels import pairwise_kernel, resolve_gamma


class _BinarySVC:
    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma=None,
        coef0=0.0,
        tol=1e-3,
        max_passes=5,
        max_iter=1000,
        random_state=None,
    ):
        self.C = float(C)
        self.kernel = kernel
        self.degree = int(degree)
        self.gamma = gamma
        self.coef0 = float(coef0)
        self.tol = float(tol)
        self.max_passes = int(max_passes)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.X_ = None
        self.y_ = None
        self.alphas_ = None
        self.b_ = 0.0
        self.support_vectors_ = None
        self.dual_coef_ = None
        self.kernel_matrix_ = None

    def _resolve_gamma(self, n_features):
        return resolve_gamma(self.gamma, n_features)

    def _kernel(self, X, Y):
        return pairwise_kernel(X, Y, kernel=self.kernel, gamma=self._gamma, coef0=self.coef0, degree=self.degree)

    def fit(self, X, y):
        X = check_2d_array(X)
        y = check_1d_array(y)
        X, y = check_same_rows(X, y)
        classes = np.unique(y)
        if classes.shape[0] != 2:
            raise ValueError("Binary SVC expects exactly two classes")

        y = np.where(y == classes[1], 1.0, -1.0)
        self.classes_ = classes
        self.X_ = X
        self.y_ = y
        self._gamma = self._resolve_gamma(X.shape[1])
        self.alphas_ = np.zeros(X.shape[0], dtype=float)
        self.b_ = 0.0
        self.kernel_matrix_ = self._kernel(X, X)

        rng = np.random.default_rng(self.random_state)
        passes = 0
        iters = 0

        def decision_row(i):
            return float(np.sum(self.alphas_ * self.y_ * self.kernel_matrix_[:, i]) + self.b_)

        while passes < self.max_passes and iters < self.max_iter:
            num_changed = 0
            for i in range(X.shape[0]):
                Ei = decision_row(i) - self.y_[i]
                ai = self.alphas_[i]
                yi = self.y_[i]
                if not ((yi * Ei < -self.tol and ai < self.C) or (yi * Ei > self.tol and ai > 0)):
                    continue

                j = int(rng.integers(0, X.shape[0] - 1))
                if j >= i:
                    j += 1

                Ej = decision_row(j) - self.y_[j]
                aj_old = self.alphas_[j]
                ai_old = ai
                yj = self.y_[j]

                if yi != yj:
                    L = max(0.0, aj_old - ai_old)
                    H = min(self.C, self.C + aj_old - ai_old)
                else:
                    L = max(0.0, ai_old + aj_old - self.C)
                    H = min(self.C, ai_old + aj_old)
                if L == H:
                    continue

                kii = self.kernel_matrix_[i, i]
                kjj = self.kernel_matrix_[j, j]
                kij = self.kernel_matrix_[i, j]
                eta = 2.0 * kij - kii - kjj
                if eta >= 0:
                    continue

                aj_new = aj_old - yj * (Ei - Ej) / eta
                aj_new = float(np.clip(aj_new, L, H))
                if abs(aj_new - aj_old) < 1e-8:
                    continue

                ai_new = ai_old + yi * yj * (aj_old - aj_new)
                self.alphas_[i] = ai_new
                self.alphas_[j] = aj_new

                b1 = (
                    self.b_
                    - Ei
                    - yi * (ai_new - ai_old) * kii
                    - yj * (aj_new - aj_old) * kij
                )
                b2 = (
                    self.b_
                    - Ej
                    - yi * (ai_new - ai_old) * kij
                    - yj * (aj_new - aj_old) * kjj
                )

                if 0.0 < ai_new < self.C:
                    self.b_ = b1
                elif 0.0 < aj_new < self.C:
                    self.b_ = b2
                else:
                    self.b_ = 0.5 * (b1 + b2)

                num_changed += 1
                iters += 1
                if iters >= self.max_iter:
                    break

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

        support = self.alphas_ > 1e-8
        self.support_vectors_ = self.X_[support]
        self.dual_coef_ = (self.alphas_[support] * self.y_[support])[:, None]
        self.support_indices_ = np.flatnonzero(support)
        return self

    def decision_function(self, X):
        if self.support_vectors_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        K = self._kernel(X, self.support_vectors_)
        scores = K @ self.dual_coef_.ravel() + self.b_
        return float(scores[0]) if scores.shape[0] == 1 else scores

    def predict(self, X):
        scores = self.decision_function(X)
        pred = np.where(scores >= 0.0, self.classes_[1], self.classes_[0])
        return pred.item() if np.ndim(pred) == 0 else pred


class SVC(BaseEstimator):
    """Support vector classifier with linear, polynomial, or RBF kernels."""

    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma=None,
        coef0=0.0,
        tol=1e-3,
        max_passes=5,
        max_iter=1000,
        random_state=None,
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.random_state = random_state
        self.classes_ = None
        self.models_ = None
        self.multi_class_ = False

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y)
        X, y = check_same_rows(X, y)
        classes = np.unique(y)
        self.classes_ = classes

        if classes.shape[0] == 2:
            self.multi_class_ = False
            model = _BinarySVC(
                C=self.C,
                kernel=self.kernel,
                degree=self.degree,
                gamma=self.gamma,
                coef0=self.coef0,
                tol=self.tol,
                max_passes=self.max_passes,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            self.models_ = [model.fit(X, y)]
            self.support_vectors_ = self.models_[0].support_vectors_
            self.dual_coef_ = self.models_[0].dual_coef_
            self.intercept_ = self.models_[0].b_
            return self

        self.multi_class_ = True
        self.models_ = []
        for idx, cls in enumerate(classes):
            y_bin = np.where(y == cls, 1.0, -1.0)
            model = _BinarySVC(
                C=self.C,
                kernel=self.kernel,
                degree=self.degree,
                gamma=self.gamma,
                coef0=self.coef0,
                tol=self.tol,
                max_passes=self.max_passes,
                max_iter=self.max_iter,
                random_state=None if self.random_state is None else self.random_state + idx,
            )
            self.models_.append(model.fit(X, y_bin))
        return self

    def decision_function(self, X):
        if not self.models_:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        if self.multi_class_:
            scores = np.column_stack([model.decision_function(X) for model in self.models_])
            return scores[0] if scores.shape[0] == 1 else scores
        return self.models_[0].decision_function(X)

    def predict(self, X):
        if not self.models_:
            raise RuntimeError("Model has not been fit yet")
        if self.multi_class_:
            scores = self.decision_function(X)
            if scores.ndim == 1:
                return self.classes_[int(np.argmax(scores))]
            idx = np.argmax(scores, axis=1)
            pred = self.classes_[idx]
            return pred[0] if pred.shape[0] == 1 else pred
        pred = self.models_[0].predict(X)
        return pred

    def score(self, X, y):
        return accuracy(y, self.predict(X))
