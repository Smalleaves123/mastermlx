from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils.validation import check_2d_array


def _joint_probs(X, perplexity=30.0, tol=1e-5, max_iter=50):
    """Compute symmetric pairwise affinities p_{j|i} from input distances."""
    n = X.shape[0]
    sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2.0 * (X @ X.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    P = np.zeros((n, n), dtype=float)
    target_entropy = np.log(perplexity)

    for i in range(n):
        beta = 1.0
        beta_min, beta_max = -np.inf, np.inf
        dists_i = np.concatenate([sq_dists[i, :i], sq_dists[i, i+1:]])

        for _ in range(max_iter):
            p_i = np.exp(-dists_i * beta)
            sum_p = np.sum(p_i)
            if sum_p < 1e-15:
                p_i = np.ones_like(dists_i) / dists_i.size
                sum_p = 1.0
            H = np.log(sum_p) + beta * np.sum(dists_i * p_i) / sum_p
            diff = H - target_entropy
            if abs(diff) < tol:
                break
            if diff > 0:
                beta_min = beta
                beta = beta * 2.0 if beta_max == np.inf else (beta + beta_max) / 2.0
            else:
                beta_max = beta
                beta = beta / 2.0 if beta_min == -np.inf else (beta + beta_min) / 2.0

        p_i = p_i / sum_p
        P[i, :i] = p_i[:i]
        P[i, i+1:] = p_i[i:]

    P = (P + P.T) / (2.0 * n)
    return np.maximum(P, 1e-15)


class TSNE(BaseEstimator):
    """t-distributed Stochastic Neighbor Embedding."""

    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0,
                 n_iter=1000, early_exaggeration=4.0, random_state=None):
        self.n_components = int(n_components)
        self.perplexity = float(perplexity)
        self.learning_rate = float(learning_rate)
        self.n_iter = int(n_iter)
        self.early_exaggeration = float(early_exaggeration)
        self.random_state = random_state
        self.embedding_ = None
        self.kl_divergence_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        n = X.shape[0]
        P = _joint_probs(X, self.perplexity)

        rng = np.random.default_rng(self.random_state)
        Y = rng.normal(0, 1e-4, size=(n, self.n_components))
        dY = np.zeros_like(Y)
        iY = np.zeros_like(Y)
        gains = np.ones_like(Y)

        for it in range(self.n_iter):
            # Q distribution (t-dist with 1 dof)
            sq_dists = (np.sum(Y**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :]
                        - 2.0 * (Y @ Y.T))
            Q_num = 1.0 / (1.0 + sq_dists)
            np.fill_diagonal(Q_num, 0.0)
            Q = Q_num / np.sum(Q_num)
            Q = np.maximum(Q, 1e-15)

            # Exaggerate P early on
            if it < 250:
                PQ = self.early_exaggeration * P - Q
            else:
                PQ = P - Q

            # Gradient
            for i in range(n):
                dY[i] = 4.0 * np.sum(
                    (PQ[i, :] * Q_num[i, :])[:, None] * (Y[i] - Y), axis=0
                )

            # Momentum gains
            gains = np.where(np.sign(dY) != np.sign(iY), gains * 0.8, gains * 1.2)
            gains = np.clip(gains, 0.01, 50.0)
            iY = 0.5 * iY - self.learning_rate * (gains * dY)
            Y += iY

            # Center
            Y -= np.mean(Y, axis=0)

        self.embedding_ = Y
        self.kl_divergence_ = float(np.sum(P * np.log(P / Q)))
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).embedding_
