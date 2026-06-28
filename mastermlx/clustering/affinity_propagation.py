from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import as_2d, check_2d_array


def _pairwise_sq_dists(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.sum(diff * diff, axis=2)


class AffinityPropagation(BaseEstimator):
    """Affinity propagation clustering."""

    def __init__(self, damping=0.5, max_iter=200, convergence_iter=15, preference=None, random_state=None):
        self.damping = float(damping)
        self.max_iter = int(max_iter)
        self.convergence_iter = int(convergence_iter)
        self.preference = preference
        self.random_state = random_state
        self.labels_ = None
        self.cluster_centers_indices_ = None
        self.cluster_centers_ = None
        self.affinity_matrix_ = None
        self.n_clusters_ = 0

    def fit(self, X, y=None):
        X = check_2d_array(X)
        if not 0.5 <= self.damping < 1.0:
            raise ValueError("damping must be in [0.5, 1)")

        S = -_pairwise_sq_dists(X)
        if self.preference is None:
            pref = np.median(S)
        elif np.isscalar(self.preference):
            pref = float(self.preference)
        else:
            pref = np.asarray(self.preference, dtype=float)
            if pref.shape != (X.shape[0],):
                raise ValueError("preference must be a scalar or an array of shape (n_samples,)")
        np.fill_diagonal(S, pref)

        n = X.shape[0]
        R = np.zeros((n, n), dtype=float)
        A = np.zeros((n, n), dtype=float)
        diag_history = []

        for _ in range(self.max_iter):
            AS = A + S
            max_idx = np.argmax(AS, axis=1)
            max_val = np.max(AS, axis=1)
            AS_copy = AS.copy()
            AS_copy[np.arange(n), max_idx] = -np.inf
            second_val = np.max(AS_copy, axis=1)
            row_best = np.where(np.arange(n)[:, None] == max_idx[:, None], second_val[:, None], max_val[:, None])
            R_new = S - row_best
            R = self.damping * R + (1.0 - self.damping) * R_new

            Rp = np.maximum(R, 0.0)
            np.fill_diagonal(Rp, R.diagonal())
            A_new = np.zeros_like(A)
            sum_pos = np.sum(Rp, axis=0)
            A_new = np.minimum(0.0, R.diagonal()[None, :] + sum_pos[None, :] - Rp)
            np.fill_diagonal(A_new, sum_pos - Rp.diagonal())
            A = self.damping * A + (1.0 - self.damping) * A_new

            exemplar_mask = np.diag(A + R) > 0
            diag_history.append(exemplar_mask)
            if len(diag_history) > self.convergence_iter:
                diag_history.pop(0)
                if all(np.array_equal(diag_history[0], x) for x in diag_history[1:]):
                    break

        exemplar_mask = np.diag(A + R) > 0
        exemplars = np.flatnonzero(exemplar_mask)
        if exemplars.size == 0:
            exemplars = np.array([int(np.argmax(np.diag(A + R)))])

        labels = np.argmax(S[:, exemplars], axis=1)

        self.affinity_matrix_ = S
        self.cluster_centers_indices_ = exemplars
        self.cluster_centers_ = X[exemplars]
        self.labels_ = labels
        self.n_clusters_ = exemplars.size
        self.responsibility_ = R
        self.availability_ = A
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        if X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError("X has a different number of features than the fitted data")
        diff = X[:, None, :] - self.cluster_centers_[None, :, :]
        labels = np.argmax(-np.sum(diff * diff, axis=2), axis=1)
        return labels[0] if labels.shape[0] == 1 else labels


AP = AffinityPropagation
