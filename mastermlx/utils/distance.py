from __future__ import annotations

import numpy as np


def euclidean_distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))


def manhattan_distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sum(np.abs(a - b), axis=-1)


def minkowski_distance(a, b, p=2.0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p = float(p)
    if p <= 0:
        raise ValueError("p must be positive")
    return np.sum(np.abs(a - b) ** p, axis=-1) ** (1.0 / p)


def chebyshev_distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.max(np.abs(a - b), axis=-1)


def cosine_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    sim = np.divide(num, np.maximum(denom, 1e-12))
    return 1.0 - sim


def hamming_distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.mean(a != b, axis=-1)


def jaccard_distance(a, b):
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    inter = np.sum(a & b, axis=-1)
    union = np.sum(a | b, axis=-1)
    return np.divide(union - inter, np.maximum(union, 1e-12))


def mahalanobis_distance(a, b, VI=None):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    if VI is None:
        VI = np.eye(diff.shape[-1], dtype=float)
    VI = np.asarray(VI, dtype=float)
    if VI.ndim != 2 or VI.shape[0] != VI.shape[1]:
        raise ValueError("VI must be a square matrix")
    return np.sqrt(np.einsum("...i,ij,...j->...", diff, VI, diff))


def pairwise_distance(X, Y, metric="euclidean", p=2.0, VI=None):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")

    diff = X[:, None, :] - Y[None, :, :]
    metric = metric.lower()
    if metric in {"euclidean", "l2"}:
        return np.sqrt(np.sum(diff * diff, axis=2))
    if metric in {"manhattan", "l1"}:
        return np.sum(np.abs(diff), axis=2)
    if metric == "minkowski":
        p = float(p)
        if p <= 0:
            raise ValueError("p must be positive")
        return np.sum(np.abs(diff) ** p, axis=2) ** (1.0 / p)
    if metric == "chebyshev":
        return np.max(np.abs(diff), axis=2)
    if metric == "cosine":
        x_norm = np.linalg.norm(X, axis=1)[:, None]
        y_norm = np.linalg.norm(Y, axis=1)[None, :]
        sim = (X @ Y.T) / np.maximum(x_norm * y_norm, 1e-12)
        return 1.0 - sim
    if metric == "hamming":
        return np.mean(X[:, None, :] != Y[None, :, :], axis=2)
    if metric == "jaccard":
        Xb = X.astype(bool)
        Yb = Y.astype(bool)
        inter = np.sum(Xb[:, None, :] & Yb[None, :, :], axis=2)
        union = np.sum(Xb[:, None, :] | Yb[None, :, :], axis=2)
        return np.divide(union - inter, np.maximum(union, 1e-12))
    if metric == "mahalanobis":
        if VI is None:
            VI = np.eye(X.shape[1], dtype=float)
        VI = np.asarray(VI, dtype=float)
        if VI.ndim != 2 or VI.shape[0] != VI.shape[1]:
            raise ValueError("VI must be a square matrix")
        if VI.shape[0] != X.shape[1]:
            raise ValueError("VI must match the number of features")
        return np.sqrt(np.einsum("...i,ij,...j->...", diff, VI, diff))
    if metric == "canberra":
        num = np.abs(diff)
        den = np.abs(X[:, None, :]) + np.abs(Y[None, :, :])
        return np.sum(num / np.maximum(den, 1e-12), axis=2)
    if metric == "bray_curtis":
        num = np.sum(np.abs(diff), axis=2)
        den = np.sum(np.abs(X[:, None, :] + Y[None, :, :]), axis=2)
        return num / np.maximum(den, 1e-12)
    raise ValueError(
        "metric must be one of: euclidean, manhattan, minkowski, chebyshev, cosine, hamming, jaccard, mahalanobis, canberra, bray_curtis"
    )


def canberra_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = np.abs(a - b)
    den = np.abs(a) + np.abs(b)
    return np.sum(num / np.maximum(den, 1e-12), axis=-1)


def bray_curtis_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = np.sum(np.abs(a - b), axis=-1)
    den = np.sum(np.abs(a + b), axis=-1)
    return num / np.maximum(den, 1e-12)


def wasserstein_distance(u, v):
    """1D Wasserstein (Earth Mover's) distance between two distributions."""
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    u_sorted = np.sort(u)
    v_sorted = np.sort(v)
    # If different sizes, interpolate v onto u's quantiles
    if u.size != v.size:
        n = min(u.size, v.size)
        u_q = np.percentile(u, np.linspace(0, 100, n))
        v_q = np.percentile(v, np.linspace(0, 100, n))
        return float(np.mean(np.abs(u_q - v_q)))
    return float(np.mean(np.abs(u_sorted - v_sorted)))


euclid = euclidean_distance
manhattan = manhattan_distance
