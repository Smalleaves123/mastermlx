from __future__ import annotations

import numpy as np

try:
    from ._distance_scalar_ops import (
        bray_curtis_distance as _cy_bray_curtis_distance,
        canberra_distance as _cy_canberra_distance,
        chebyshev_distance as _cy_chebyshev_distance,
        cosine_distance as _cy_cosine_distance,
        euclidean_distance as _cy_euclidean_distance,
        hamming_distance as _cy_hamming_distance,
        jaccard_distance as _cy_jaccard_distance,
        mahalanobis_distance as _cy_mahalanobis_distance,
        manhattan_distance as _cy_manhattan_distance,
        minkowski_distance as _cy_minkowski_distance,
    )
except ImportError:  # pragma: no cover - fallback when Cython extensions are unavailable
    _cy_bray_curtis_distance = None
    _cy_canberra_distance = None
    _cy_chebyshev_distance = None
    _cy_cosine_distance = None
    _cy_euclidean_distance = None
    _cy_hamming_distance = None
    _cy_jaccard_distance = None
    _cy_mahalanobis_distance = None
    _cy_manhattan_distance = None
    _cy_minkowski_distance = None


def euclidean_distance(a, b):
    if _cy_euclidean_distance is not None:
        return _cy_euclidean_distance(a, b)
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))


def manhattan_distance(a, b):
    if _cy_manhattan_distance is not None:
        return _cy_manhattan_distance(a, b)
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sum(np.abs(a - b), axis=-1)


def minkowski_distance(a, b, p=2.0):
    if _cy_minkowski_distance is not None:
        return _cy_minkowski_distance(a, b, float(p))
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p = float(p)
    if p <= 0:
        raise ValueError("p must be positive")
    return np.sum(np.abs(a - b) ** p, axis=-1) ** (1.0 / p)


def chebyshev_distance(a, b):
    if _cy_chebyshev_distance is not None:
        return _cy_chebyshev_distance(a, b)
    a = np.asarray(a)
    b = np.asarray(b)
    return np.max(np.abs(a - b), axis=-1)


def cosine_distance(a, b):
    if _cy_cosine_distance is not None:
        return _cy_cosine_distance(a, b)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    sim = np.divide(num, np.maximum(denom, 1e-12))
    return 1.0 - sim


def hamming_distance(a, b):
    if _cy_hamming_distance is not None:
        return _cy_hamming_distance(a, b)
    a = np.asarray(a)
    b = np.asarray(b)
    return np.mean(a != b, axis=-1)


def jaccard_distance(a, b):
    if _cy_jaccard_distance is not None:
        return _cy_jaccard_distance(a, b)
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    inter = np.sum(a & b, axis=-1)
    union = np.sum(a | b, axis=-1)
    return np.divide(union - inter, np.maximum(union, 1e-12))


def mahalanobis_distance(a, b, VI=None):
    if _cy_mahalanobis_distance is not None:
        return _cy_mahalanobis_distance(a, b, VI=VI)
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
    """Pairwise distance matrix between X (n×d) and Y (m×d).

    Uses C++ acceleration when available for euclidean, manhattan, chebyshev,
    minkowski, canberra, and bray_curtis — avoiding the large 3D intermediate
    array that the pure-NumPy fallback creates.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")

    metric = metric.lower()
    if metric == "minkowski":
        p = float(p)
        if p <= 0:
            raise ValueError("p must be positive")

    # ---- C++ / accel fast paths (no 3D intermediate) ----
    if metric in {"euclidean", "l2"}:
        from ..accel import pairwise_distances as _cpp_euc
        return _cpp_euc(X, Y)

    if metric in {"manhattan", "l1"}:
        from ..accel import pairwise_manhattan_distances as _cpp_man
        return _cpp_man(X, Y)

    if metric == "chebyshev":
        from ..accel import pairwise_chebyshev as _cpp_ch
        return _cpp_ch(X, Y)

    if metric == "minkowski":
        from ..accel import pairwise_minkowski as _cpp_mk
        return _cpp_mk(X, Y, p)

    if metric == "canberra":
        from ..accel import pairwise_canberra as _cpp_ca
        return _cpp_ca(X, Y)

    if metric == "bray_curtis":
        from ..accel import pairwise_bray_curtis as _cpp_bc
        return _cpp_bc(X, Y)

    # ---- Metrics that are already memory-efficient (no 3D overhead) ----
    if metric == "cosine":
        from ..accel import pairwise_cosine_distances as _cy_cosine
        return _cy_cosine(X, Y)

    if metric == "hamming":
        from ..accel import pairwise_hamming_distances as _cy_hamming
        return _cy_hamming(X, Y)

    if metric == "jaccard":
        from ..accel import pairwise_jaccard_distances as _cy_jaccard
        return _cy_jaccard(X, Y)

    if metric == "mahalanobis":
        from ..accel import pairwise_mahalanobis_distances as _cy_maha
        if VI is None:
            VI = np.eye(X.shape[1], dtype=float)
        VI = np.asarray(VI, dtype=float)
        if VI.ndim != 2 or VI.shape[0] != VI.shape[1]:
            raise ValueError("VI must be a square matrix")
        if VI.shape[0] != X.shape[1]:
            raise ValueError("VI must match the number of features")
        return _cy_maha(X, Y, VI)

    raise ValueError(
        "metric must be one of: euclidean, manhattan, minkowski, chebyshev, "
        "cosine, hamming, jaccard, mahalanobis, canberra, bray_curtis"
    )


def canberra_distance(a, b):
    if _cy_canberra_distance is not None:
        return _cy_canberra_distance(a, b)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = np.abs(a - b)
    den = np.abs(a) + np.abs(b)
    return np.sum(num / np.maximum(den, 1e-12), axis=-1)


def bray_curtis_distance(a, b):
    if _cy_bray_curtis_distance is not None:
        return _cy_bray_curtis_distance(a, b)
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
