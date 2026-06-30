from __future__ import annotations

import importlib
from functools import lru_cache

import numpy as np

from ..config import get_backend


# ============================================================================
#  NumPy fallback implementations (kept here for reference / benchmarking)
# ============================================================================

def _numpy_pairwise_squared_euclidean(X, Y):
    diff = X[:, None, :] - Y[None, :, :]
    return np.sum(diff * diff, axis=2)


def _numpy_pairwise_distances(X, Y):
    return np.sqrt(np.maximum(_numpy_pairwise_squared_euclidean(X, Y), 0.0))


def _numpy_pairwise_manhattan(X, Y):
    diff = np.abs(X[:, None, :] - Y[None, :, :])
    return np.sum(diff, axis=2)


def _numpy_pairwise_chebyshev(X, Y):
    return np.max(np.abs(X[:, None, :] - Y[None, :, :]), axis=2)


def _numpy_pairwise_minkowski(X, Y, p):
    diff = np.abs(X[:, None, :].astype(float) - Y[None, :, :].astype(float))
    return np.sum(diff ** p, axis=2) ** (1.0 / p)


def _numpy_pairwise_canberra(X, Y):
    diff = np.abs(X[:, None, :] - Y[None, :, :])
    den = np.abs(X[:, None, :]) + np.abs(Y[None, :, :])
    return np.sum(diff / np.maximum(den, 1e-12), axis=2)


def _numpy_pairwise_bray_curtis(X, Y):
    num = np.sum(np.abs(X[:, None, :] - Y[None, :, :]), axis=2)
    den = np.sum(np.abs(X[:, None, :] + Y[None, :, :]), axis=2)
    return num / np.maximum(den, 1e-12)


# ============================================================================
#  Backend loaders (lazy, cached)
# ============================================================================

@lru_cache(maxsize=1)
def _load_cpp_backend():
    try:
        return importlib.import_module("mastermlx.accel._distance_cpp")
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _load_cython_backend():
    try:
        return importlib.import_module("mastermlx.accel._distance_ops")
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _load_cpp_kernels():
    try:
        return importlib.import_module("mastermlx.accel._kernels_cpp")
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _load_cython_tree():
    try:
        return importlib.import_module("mastermlx.accel._tree_ops")
    except ImportError:
        return None


def get_active_backend():
    preferred = get_backend()
    if preferred == "numpy":
        return "numpy"
    cython_mod = _load_cython_backend()
    if preferred == "cython":
        if cython_mod is None:
            raise RuntimeError("Cython backend requested but compiled extensions are unavailable")
        return "cython"
    return "cython" if cython_mod is not None else "numpy"


# ============================================================================
#  Public API — pairwise distances  (C++ > Cython > NumPy)
# ============================================================================

def pairwise_squared_euclidean(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    cpp = _load_cpp_backend()
    if cpp is not None:
        return cpp.pairwise_squared_euclidean(X, Y)
    backend = get_active_backend()
    if backend == "cython":
        mod = _load_cython_backend()
        if mod is not None:
            return mod.pairwise_squared_euclidean(X, Y)
    return _numpy_pairwise_squared_euclidean(X, Y)


def pairwise_distances(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    cpp = _load_cpp_backend()
    if cpp is not None:
        return cpp.pairwise_distances(X, Y)
    mod = _load_cython_backend()
    if mod is not None:
        return mod.pairwise_distances(X, Y)
    return _numpy_pairwise_distances(X, Y)


def pairwise_manhattan_distances(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    cpp = _load_cpp_backend()
    if cpp is not None:
        return cpp.pairwise_manhattan_distances(X, Y)
    mod = _load_cython_backend()
    if mod is not None:
        return mod.pairwise_manhattan_distances(X, Y)
    return _numpy_pairwise_manhattan(X, Y)


def pairwise_cosine_distances(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    mod = _load_cython_backend()
    if mod is not None:
        return mod.pairwise_cosine_distances(X, Y)
    x_norm = np.linalg.norm(X, axis=1)[:, None]
    y_norm = np.linalg.norm(Y, axis=1)[None, :]
    sim = (X @ Y.T) / np.maximum(x_norm * y_norm, 1e-12)
    return 1.0 - sim


def pairwise_hamming_distances(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    mod = _load_cython_backend()
    if mod is not None:
        return mod.pairwise_hamming_distances(X, Y)
    return np.mean(X[:, None, :] != Y[None, :, :], axis=2)


def pairwise_jaccard_distances(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    mod = _load_cython_backend()
    if mod is not None:
        return mod.pairwise_jaccard_distances(X, Y)
    Xb = X.astype(bool)
    Yb = Y.astype(bool)
    inter = np.sum(Xb[:, None, :] & Yb[None, :, :], axis=2)
    union = np.sum(Xb[:, None, :] | Yb[None, :, :], axis=2)
    return np.divide(union - inter, np.maximum(union, 1e-12))


def pairwise_mahalanobis_distances(X, Y, VI=None):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if VI is None:
        VI = np.eye(X.shape[1], dtype=float)
    VI = np.asarray(VI, dtype=float)
    mod = _load_cython_backend()
    if mod is not None:
        return mod.pairwise_mahalanobis_distances(X, Y, VI)
    diff = X[:, None, :] - Y[None, :, :]
    return np.sqrt(np.einsum("...i,ij,...j->...", diff, VI, diff))


def pairwise_chebyshev(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    cpp = _load_cpp_backend()
    if cpp is not None:
        return cpp.pairwise_chebyshev(X, Y)
    return _numpy_pairwise_chebyshev(X, Y)


def pairwise_minkowski(X, Y, p=2.0):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    p = float(p)
    if p <= 0:
        raise ValueError("p must be positive")
    cpp = _load_cpp_backend()
    if cpp is not None:
        return cpp.pairwise_minkowski(X, Y, p)
    return _numpy_pairwise_minkowski(X, Y, p)


def pairwise_canberra(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    cpp = _load_cpp_backend()
    if cpp is not None:
        return cpp.pairwise_canberra(X, Y)
    return _numpy_pairwise_canberra(X, Y)


def pairwise_bray_curtis(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    cpp = _load_cpp_backend()
    if cpp is not None:
        return cpp.pairwise_bray_curtis(X, Y)
    return _numpy_pairwise_bray_curtis(X, Y)


# ============================================================================
#  Public API — kernels  (C++ > NumPy)
# ============================================================================

def cpp_rbf_kernel(X, Y, gamma):
    """RBF kernel with C++ acceleration (avoids 3D intermediate array)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    cpp = _load_cpp_kernels()
    if cpp is not None:
        return cpp.rbf_kernel(X, Y, float(gamma))
    # NumPy fallback: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    x2 = np.sum(X ** 2, axis=1)[:, None]
    y2 = np.sum(Y ** 2, axis=1)[None, :]
    d2 = np.maximum(x2 + y2 - 2.0 * (X @ Y.T), 0.0)
    return np.exp(-float(gamma) * d2)


def cpp_rbf_kernel_fast(X, Y, xn2, yn2, gamma):
    """RBF kernel using pre-computed squared norms (fastest path)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    cpp = _load_cpp_kernels()
    if cpp is not None:
        return cpp.rbf_kernel_fast(X, Y,
                                   np.asarray(xn2, dtype=float),
                                   np.asarray(yn2, dtype=float),
                                   float(gamma))
    d2 = np.maximum(xn2[:, None] + yn2[None, :] - 2.0 * (X @ Y.T), 0.0)
    return np.exp(-float(gamma) * d2)


def cpp_laplacian_kernel(X, Y, gamma):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    cpp = _load_cpp_kernels()
    if cpp is not None:
        return cpp.laplacian_kernel(X, Y, float(gamma))
    l1 = np.sum(np.abs(X[:, None, :] - Y[None, :, :]), axis=2)
    return np.exp(-float(gamma) * l1)


def cpp_chi2_kernel(X, Y, gamma):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if np.any(X < 0) or np.any(Y < 0):
        raise ValueError("chi2_kernel expects non-negative inputs")
    cpp = _load_cpp_kernels()
    if cpp is not None:
        return cpp.chi2_kernel(X, Y, float(gamma))
    num = (X[:, None, :] - Y[None, :, :]) ** 2
    den = X[:, None, :] + Y[None, :, :] + 1e-12
    chi2 = 0.5 * np.sum(num / den, axis=2)
    return np.exp(-float(gamma) * chi2)


def cpp_additive_chi2_kernel(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if np.any(X < 0) or np.any(Y < 0):
        raise ValueError("additive_chi2_kernel expects non-negative inputs")
    cpp = _load_cpp_kernels()
    if cpp is not None:
        return cpp.additive_chi2_kernel(X, Y)
    num = 2.0 * X[:, None, :] * Y[None, :, :]
    den = X[:, None, :] + Y[None, :, :] + 1e-12
    return np.sum(num / den, axis=2)


def cpp_hellinger_kernel(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if np.any(X < 0) or np.any(Y < 0):
        raise ValueError("hellinger_kernel expects non-negative inputs")
    cpp = _load_cpp_kernels()
    if cpp is not None:
        return cpp.hellinger_kernel(X, Y)
    return np.sum(np.sqrt(X)[:, None, :] * np.sqrt(Y)[None, :, :], axis=2)


# ============================================================================
#  Tree split helpers
# ============================================================================

def best_split_classifier(X, y, min_samples_leaf):
    mod = _load_cython_tree()
    if mod is not None:
        return mod._best_split_classifier(X, y, min_samples_leaf)
    return None


def best_split_regressor(X, y, min_samples_leaf):
    mod = _load_cython_tree()
    if mod is not None:
        return mod._best_split_regressor(X, y, min_samples_leaf)
    return None


# ============================================================================
#  Aliases
# ============================================================================
active_backend = get_active_backend
pairwise_sq_euclid = pairwise_squared_euclidean
pairwise_manhattan = pairwise_manhattan_distances
pairwise_dist = pairwise_distances
