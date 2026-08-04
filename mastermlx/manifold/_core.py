from __future__ import annotations

import numpy as np

from ..accel.backends import pairwise_distances
from ..config import get_backend
from ..graphs.csr import _load_cpp
from ..utils import check_2d_array


def pairwise_dist(X):
    X = check_2d_array(X).astype(float)
    return pairwise_distances(X, X)


def kgraph(D, k):
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if k < 1 or k >= n:
        raise ValueError("n_neighbors must be between 1 and n_samples - 1")

    A = np.full_like(D, np.inf)
    distances = D.copy()
    distances[np.diag_indices(n)] = np.inf
    order = np.argsort(distances, axis=1, kind="stable")[:, :k]
    for i in range(n):
        A[i, order[i]] = D[i, order[i]]
    A = np.minimum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A


def all_pairs_shortest(W):
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix")
    n = W.shape[0]
    cpp = _load_cpp(get_backend())
    cpp_all_pairs = getattr(cpp, "all_pairs_dijkstra", None) if cpp is not None else None
    if callable(cpp_all_pairs) and np.all(W >= 0.0):
        connected = np.isfinite(W)
        connected[np.diag_indices(n)] = False
        rows, columns = np.nonzero(connected)
        indptr = np.bincount(rows, minlength=n).cumsum()
        indptr = np.concatenate(([0], indptr)).astype(np.int64, copy=False)
        distances = cpp_all_pairs(
            indptr,
            columns.astype(np.int64, copy=False),
            W[rows, columns].astype(float, copy=False),
        )
        if not np.isfinite(distances).all():
            raise ValueError("graph must be connected")
        return distances
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
