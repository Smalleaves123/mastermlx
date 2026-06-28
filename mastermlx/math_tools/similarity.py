from __future__ import annotations

import numpy as np


def _rank(x):
    """Rank 1D array with tie averaging, matches scipy.stats.rankdata."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(1.0, x.size + 1.0)
    # Average ranks within tied groups
    uniq, inv = np.unique(x, return_inverse=True)
    for g in range(uniq.size):
        mask = inv == g
        ranks[mask] = np.mean(ranks[mask])
    return ranks


def cosine_sim(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide(num, np.maximum(den, 1e-12))


def dot_sim(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sum(a * b, axis=-1)


def pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D")
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    den = np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2))
    return float(np.dot(xc, yc) / den) if den > 0 else 0.0


def spearman_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D")
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    rx = _rank(x)
    ry = _rank(y)
    return pearson_r(rx, ry)


def kendall_tau(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D")
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    n = x.size
    if n < 2:
        return 1.0
    conc = 0
    disc = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dy > 0:
                conc += 1
            elif dx * dy < 0:
                disc += 1
    total = conc + disc
    return float((conc - disc) / total) if total > 0 else 0.0


def pairwise_cosine(X, Y=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if Y is None:
        Y = X
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be 2D")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")
    xn = np.linalg.norm(X, axis=1)[:, None]
    yn = np.linalg.norm(Y, axis=1)[None, :]
    return (X @ Y.T) / np.maximum(xn * yn, 1e-12)
