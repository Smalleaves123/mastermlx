from __future__ import annotations

import numpy as np

from ..utils import check_1d_array, check_2d_array


def pairwise_sqdist(X, Y=None):
    X = check_2d_array(X).astype(float)
    if Y is None:
        Y = X
    else:
        Y = check_2d_array(Y).astype(float)
    x2 = np.sum(X ** 2, axis=1, keepdims=True)
    y2 = np.sum(Y ** 2, axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2.0 * (X @ Y.T), 0.0)


def rbf_affinity(X, gamma):
    D2 = pairwise_sqdist(X)
    A = np.exp(-gamma * D2)
    np.fill_diagonal(A, 0.0)
    return A


def knn_affinity(X, n_neighbors):
    X = check_2d_array(X).astype(float)
    n = X.shape[0]
    if n_neighbors < 1 or n_neighbors >= n:
        raise ValueError("n_neighbors must be between 1 and n_samples - 1")
    D2 = pairwise_sqdist(X)
    A = np.zeros((n, n), dtype=float)
    nbr = np.argsort(D2, axis=1)[:, 1 : n_neighbors + 1]
    for i in range(n):
        A[i, nbr[i]] = 1.0
    A = np.maximum(A, A.T)
    return A


def row_norm(A):
    A = np.asarray(A, dtype=float)
    s = np.sum(A, axis=1, keepdims=True)
    s = np.where(s == 0.0, 1.0, s)
    return A / s


def sym_norm(A):
    A = np.asarray(A, dtype=float)
    d = np.sum(A, axis=1)
    d = np.where(d == 0.0, 1.0, d)
    s = 1.0 / np.sqrt(d)
    return (s[:, None] * A) * s[None, :]


def make_y(y):
    y = check_1d_array(y, name="y")
    classes = np.unique(y[y != -1])
    if classes.size == 0:
        raise ValueError("y must contain at least one labeled sample")
    return y, classes


def one_hot(y, classes):
    y = np.asarray(y)
    out = np.zeros((y.shape[0], classes.shape[0]), dtype=float)
    for i, cls in enumerate(classes):
        out[y == cls, i] = 1.0
    return out


def hard_labels(F, classes):
    return classes[np.argmax(F, axis=1)]

