from __future__ import annotations

import numpy as np

from ..utils import check_2d_array


def pairwise_dist(X):
    X = check_2d_array(X).astype(float)
    x2 = np.sum(X ** 2, axis=1, keepdims=True)
    d2 = np.maximum(x2 + x2.T - 2.0 * (X @ X.T), 0.0)
    return np.sqrt(d2)


def kgraph(D, k):
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if k < 1 or k >= n:
        raise ValueError("n_neighbors must be between 1 and n_samples - 1")

    A = np.full_like(D, np.inf)
    order = np.argsort(D, axis=1)[:, 1 : k + 1]
    for i in range(n):
        A[i, order[i]] = D[i, order[i]]
    A = np.minimum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A


def all_pairs_shortest(W):
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    D = W.copy()
    for k in range(n):
        via = D[:, [k]] + D[[k], :]
        D = np.minimum(D, via)
    if not np.isfinite(D).all():
        raise ValueError("graph must be connected")
    return D


def center_dist(D):
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    D2 = D ** 2
    return -0.5 * J @ D2 @ J


def eig_embed(M, k, high=True, scale=True):
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(vals)
    if high:
        order = order[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    if high:
        keep = vals > 1e-12
        vals = vals[keep]
        vecs = vecs[:, keep]
    else:
        keep = vals > 1e-12 if np.any(vals > 1e-12) else np.ones_like(vals, dtype=bool)
        vals = vals[keep]
        vecs = vecs[:, keep]
    if vals.size == 0:
        raise ValueError("embedding failed because the matrix is numerically rank deficient")
    k = min(int(k), vals.size)
    vals = vals[:k]
    vecs = vecs[:, :k]
    if scale:
        vecs = vecs * np.sqrt(vals)[None, :]
    return vecs, vals
