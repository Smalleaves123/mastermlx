from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import check_2d_array
from ._core import hard_labels, knn_affinity, make_y, one_hot, rbf_affinity, row_norm, sym_norm


class _LabelBase(BaseEstimator):
    def __init__(self, kernel="rbf", gamma=None, n_neighbors=5, max_iter=1000, tol=1e-4):
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = int(n_neighbors)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.X_ = None
        self.y_ = None
        self.classes_ = None
        self.label_distributions_ = None
        self.transduction_ = None
        self.n_iter_ = 0

    def _resolve_gamma(self, n_features):
        if self.gamma is None:
            return 1.0 / max(n_features, 1)
        if self.gamma == "scale":
            return 1.0 / max(n_features, 1)
        return float(self.gamma)

    def _affinity(self, X):
        if self.kernel == "rbf":
            return rbf_affinity(X, self._gamma)
        if self.kernel == "knn":
            return knn_affinity(X, self.n_neighbors)
        raise ValueError("kernel must be one of: rbf, knn")

    def _resolve_labels(self, y):
        y, classes = make_y(y)
        Y = one_hot(y, classes)
        labeled = y != -1
        return y, classes, Y, labeled

    def predict(self, X=None):
        if self.transduction_ is None:
            raise RuntimeError("Model has not been fit yet")
        if X is None:
            return self.transduction_
        X = check_2d_array(X).astype(float)
        if self.X_ is None or X.shape != self.X_.shape or not np.allclose(X, self.X_):
            raise ValueError("This model predicts labels for the fitted samples only")
        return self.transduction_

    def predict_proba(self, X=None):
        if self.label_distributions_ is None:
            raise RuntimeError("Model has not been fit yet")
        if X is None:
            return self.label_distributions_
        X = check_2d_array(X).astype(float)
        if self.X_ is None or X.shape != self.X_.shape or not np.allclose(X, self.X_):
            raise ValueError("This model predicts probabilities for the fitted samples only")
        return self.label_distributions_

    def fit_predict(self, X, y):
        return self.fit(X, y).predict()


class LabelPropagation(_LabelBase):
    """Label propagation on a graph built from the data."""

    def fit(self, X, y):
        X = check_2d_array(X).astype(float)
        y, classes, Y, labeled = self._resolve_labels(y)
        self.X_ = X
        self.y_ = y
        self.classes_ = classes
        self._gamma = self._resolve_gamma(X.shape[1])

        A = self._affinity(X)
        S = row_norm(A)
        F = Y.copy()
        F[~labeled] = 0.0

        for it in range(1, self.max_iter + 1):
            prev = F.copy()
            F = S @ F
            F[labeled] = Y[labeled]
            delta = np.max(np.abs(F - prev))
            if delta < self.tol:
                self.n_iter_ = it
                break
        else:
            self.n_iter_ = self.max_iter

        F = row_norm(F)
        self.label_distributions_ = F
        self.transduction_ = hard_labels(F, self.classes_)
        return self


class LabelSpreading(_LabelBase):
    """Label spreading with soft clamping."""

    def __init__(self, kernel="rbf", gamma=None, n_neighbors=5, alpha=0.2, max_iter=1000, tol=1e-4):
        super().__init__(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, max_iter=max_iter, tol=tol)
        self.alpha = float(alpha)

    def fit(self, X, y):
        X = check_2d_array(X).astype(float)
        y, classes, Y, labeled = self._resolve_labels(y)
        self.X_ = X
        self.y_ = y
        self.classes_ = classes
        self._gamma = self._resolve_gamma(X.shape[1])

        A = self._affinity(X)
        S = sym_norm(A)
        F = Y.copy()
        Y0 = Y.copy()
        Y0[~labeled] = 0.0

        alpha = float(self.alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")

        for it in range(1, self.max_iter + 1):
            prev = F.copy()
            F = alpha * (S @ F) + (1.0 - alpha) * Y0
            F[labeled] = (1.0 - alpha) * Y[labeled] + alpha * F[labeled]
            delta = np.max(np.abs(F - prev))
            if delta < self.tol:
                self.n_iter_ = it
                break
        else:
            self.n_iter_ = self.max_iter

        F = row_norm(F)
        self.label_distributions_ = F
        self.transduction_ = hard_labels(F, self.classes_)
        return self
