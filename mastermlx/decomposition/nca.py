from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils.validation import check_1d_array, check_2d_array, check_same_rows


class NCA(BaseTransformer):
    """Neighborhood Component Analysis — learns a linear transformation
    that maximizes k-NN leave-one-out classification accuracy.

    Parameters
    ----------
    n_components : int
        Output dimensionality.
    lr : float
        Learning rate.
    max_iter : int
        Maximum gradient descent iterations.
    tol : float
        Convergence tolerance.
    random_state : int or None
        Random seed for initialization.
    """

    def __init__(self, n_components=None, lr=0.01, max_iter=500, tol=1e-6,
                 random_state=None):
        self.n_components = n_components
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.A_ = None
        self.loss_ = []

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y)
        X, y = check_same_rows(X, y)
        n, d = X.shape
        out_dim = self.n_components or d

        rng = np.random.default_rng(self.random_state)
        self.A_ = rng.normal(0, 0.01, size=(d, out_dim))
        self.loss_ = []

        prev_loss = np.inf
        for _ in range(self.max_iter):
            # Transformed data
            Z = X @ self.A_
            # Pairwise squared distances in transformed space
            sq = np.sum(Z**2, axis=1)[:, None] + np.sum(Z**2, axis=1)[None, :] \
                 - 2.0 * (Z @ Z.T)
            sq = np.maximum(sq, 0.0)
            # Softmax probabilities (exclude self)
            p = np.exp(-sq)
            np.fill_diagonal(p, 0.0)
            row_sum = p.sum(axis=1, keepdims=True)
            row_sum = np.maximum(row_sum, 1e-15)
            p /= row_sum

            # Prob correct classification per point
            same_class = (y[:, None] == y[None, :]).astype(float)
            np.fill_diagonal(same_class, 0.0)
            pi = np.sum(p * same_class, axis=1)
            loss = -np.mean(np.log(np.maximum(pi, 1e-15)))
            self.loss_.append(loss)

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

            # Gradient
            grad = np.zeros_like(self.A_)
            p_same = p * same_class
            for i in range(n):
                if pi[i] < 1e-15:
                    continue
                diff_i = Z[i] - Z
                weighted = (p_same[i, :, None] - pi[i] * p[i, :, None]) * diff_i
                grad += X[i, :, None] * weighted.sum(axis=0)[None, :]

            grad /= n
            self.A_ -= self.lr * grad

        return self

    def transform(self, X):
        X = check_2d_array(X).astype(float)
        if self.A_ is None:
            raise RuntimeError("Model has not been fit yet")
        return X @ self.A_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
