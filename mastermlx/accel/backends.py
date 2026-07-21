from __future__ import annotations

import importlib
from functools import lru_cache

import numpy as np

from ..config import get_backend
from ._validate import float_array


_CYTHON_API = (
    "pairwise_squared_euclidean",
    "pairwise_distances",
    "pairwise_manhattan_distances",
    "pairwise_cosine_distances",
    "pairwise_hamming_distances",
    "pairwise_jaccard_distances",
    "pairwise_mahalanobis_distances",
)
_CPP_API = (
    "pairwise_squared_euclidean",
    "pairwise_distances",
    "pairwise_manhattan_distances",
    "pairwise_chebyshev",
    "pairwise_minkowski",
    "pairwise_canberra",
    "pairwise_bray_curtis",
)
_KERNEL_API = (
    "rbf_kernel",
    "rbf_kernel_fast",
    "laplacian_kernel",
    "chi2_kernel",
    "additive_chi2_kernel",
    "hellinger_kernel",
)


def _has_api(module, names):
    return module is not None and all(callable(getattr(module, name, None)) for name in names)


# ============================================================================
#  NumPy fallback implementations (kept here for reference / benchmarking)
# ============================================================================

def _numpy_pairwise_squared_euclidean(X, Y):
    """Compute squared Euclidean distances without a 3D broadcast buffer."""
    # ||x-y||² = ||x||² + ||y||² - 2 x·y.  Besides allowing BLAS to handle
    # the expensive part, this keeps memory at O(nm) instead of O(nmd).
    x_norms = np.einsum("ij,ij->i", X, X)[:, None]
    y_norms = np.einsum("ij,ij->i", Y, Y)[None, :]
    distances = x_norms + y_norms - 2.0 * (X @ Y.T)
    return np.maximum(distances, 0.0)


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


def _pairwise_inputs(X, Y):
    X = float_array(X, 2, "X")
    Y = float_array(Y, 2, "Y")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")
    return X, Y


# ============================================================================
#  Backend loaders (lazy and backend-aware)
# ============================================================================

@lru_cache(maxsize=1)
def _import_cpp_backend():
    try:
        return importlib.import_module("mastermlx.accel._distance_cpp")
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _import_cython_backend():
    try:
        return importlib.import_module("mastermlx.accel._distance_ops")
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _import_cpp_kernels():
    try:
        return importlib.import_module("mastermlx.accel._kernels_cpp")
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _import_cpp_signal():
    try:
        return importlib.import_module("mastermlx.accel._signal_cpp")
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _import_cython_tree():
    try:
        return importlib.import_module("mastermlx.accel._tree_ops")
    except ImportError:
        return None


def _load_cpp_backend():
    module = _import_cpp_backend() if get_backend() == "auto" else None
    return module if _has_api(module, _CPP_API) else None


def _load_cython_backend():
    module = _import_cython_backend() if get_backend() in {"auto", "cython"} else None
    return module if _has_api(module, _CYTHON_API) else None


def _load_cpp_kernels():
    module = _import_cpp_kernels() if get_backend() == "auto" else None
    return module if _has_api(module, _KERNEL_API) else None


def _load_cpp_signal():
    module = _import_cpp_signal() if get_backend() == "auto" else None
    return module if _has_api(module, ("frame_signal", "iir_filter_1d", "online_cusum")) else None


def _load_cython_tree():
    module = _import_cython_tree() if get_backend() in {"auto", "cython"} else None
    return module if _has_api(module, ("_best_split_classifier", "_best_split_regressor")) else None


def _load_optional_module(name, api):
    if get_backend() == "numpy":
        return None
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    return module if _has_api(module, api) else None


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


def backend_report():
    """Return compiled-backend availability and the active backend."""

    return {
        "requested": get_backend(),
        "active": get_active_backend(),
        "cython": _load_cython_backend() is not None,
        "cpp_distance": _load_cpp_backend() is not None,
        "cpp_kernels": _load_cpp_kernels() is not None,
        "cpp_signal": _load_cpp_signal() is not None,
        "cython_tree": _load_cython_tree() is not None,
        "cnn": _load_optional_module(
            "mastermlx.accel._cnn_ops",
            ("im2col", "col2im", "maxpool_forward", "maxpool_backward"),
        ) is not None,
        "conv1d": _load_optional_module(
            "mastermlx.accel._conv1d_ops", ("im2col1d", "col2im1d")
        ) is not None,
        "rnn": _load_optional_module(
            "mastermlx.accel._rnn_ops", ("simple_rnn_forward", "lstm_forward", "gru_forward")
        ) is not None,
        "signal": _load_optional_module(
            "mastermlx.accel._signal_ops", ("iir_filter_1d", "ridge_path")
        ) is not None,
    }


# ============================================================================
#  Public API — pairwise distances  (C++ > Cython > NumPy)
# ============================================================================

def pairwise_squared_euclidean(X, Y):
    X, Y = _pairwise_inputs(X, Y)
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
    X, Y = _pairwise_inputs(X, Y)
    cpp = _load_cpp_backend()
    if cpp is not None:
        return cpp.pairwise_distances(X, Y)
    mod = _load_cython_backend()
    if mod is not None:
        return mod.pairwise_distances(X, Y)
    return _numpy_pairwise_distances(X, Y)


def pairwise_manhattan_distances(X, Y):
    X, Y = _pairwise_inputs(X, Y)
    cpp = _load_cpp_backend()
    if cpp is not None:
        return cpp.pairwise_manhattan_distances(X, Y)
    mod = _load_cython_backend()
    if mod is not None:
        return mod.pairwise_manhattan_distances(X, Y)
    return _numpy_pairwise_manhattan(X, Y)


def pairwise_cosine_distances(X, Y):
    X, Y = _pairwise_inputs(X, Y)
    mod = _load_cython_backend()
    if mod is not None:
        return mod.pairwise_cosine_distances(X, Y)
    x_norm = np.linalg.norm(X, axis=1)[:, None]
    y_norm = np.linalg.norm(Y, axis=1)[None, :]
    sim = (X @ Y.T) / np.maximum(x_norm * y_norm, 1e-12)
    return 1.0 - sim


def pairwise_hamming_distances(X, Y):
    X, Y = _pairwise_inputs(X, Y)
    mod = _load_cython_backend()
    if mod is not None:
        return mod.pairwise_hamming_distances(X, Y)
    return np.mean(X[:, None, :] != Y[None, :, :], axis=2)


def pairwise_jaccard_distances(X, Y):
    X, Y = _pairwise_inputs(X, Y)
    mod = _load_cython_backend()
    if mod is not None:
        return mod.pairwise_jaccard_distances(X, Y)
    Xb = X.astype(bool)
    Yb = Y.astype(bool)
    inter = np.sum(Xb[:, None, :] & Yb[None, :, :], axis=2)
    union = np.sum(Xb[:, None, :] | Yb[None, :, :], axis=2)
    return np.divide(union - inter, np.maximum(union, 1e-12))


def pairwise_mahalanobis_distances(X, Y, VI=None):
    X, Y = _pairwise_inputs(X, Y)
    if VI is None:
        VI = np.eye(X.shape[1], dtype=float)
    VI = np.asarray(VI, dtype=float)
    if VI.shape != (X.shape[1], X.shape[1]) or not np.isfinite(VI).all():
        raise ValueError("VI must be a finite square matrix matching the feature count")
    mod = _load_cython_backend()
    if mod is not None:
        return mod.pairwise_mahalanobis_distances(X, Y, VI)
    diff = X[:, None, :] - Y[None, :, :]
    return np.sqrt(np.einsum("...i,ij,...j->...", diff, VI, diff))


def pairwise_chebyshev(X, Y):
    X, Y = _pairwise_inputs(X, Y)
    cpp = _load_cpp_backend()
    if cpp is not None:
        return cpp.pairwise_chebyshev(X, Y)
    return _numpy_pairwise_chebyshev(X, Y)


def pairwise_minkowski(X, Y, p=2.0):
    X, Y = _pairwise_inputs(X, Y)
    p = float(p)
    if not np.isfinite(p) or p <= 0:
        raise ValueError("p must be positive")
    cpp = _load_cpp_backend()
    if cpp is not None:
        return cpp.pairwise_minkowski(X, Y, p)
    return _numpy_pairwise_minkowski(X, Y, p)


def pairwise_canberra(X, Y):
    X, Y = _pairwise_inputs(X, Y)
    cpp = _load_cpp_backend()
    if cpp is not None:
        return cpp.pairwise_canberra(X, Y)
    return _numpy_pairwise_canberra(X, Y)


def pairwise_bray_curtis(X, Y):
    X, Y = _pairwise_inputs(X, Y)
    cpp = _load_cpp_backend()
    if cpp is not None:
        return cpp.pairwise_bray_curtis(X, Y)
    return _numpy_pairwise_bray_curtis(X, Y)


# ============================================================================
#  Public API — kernels  (C++ > NumPy)
# ============================================================================

def cpp_rbf_kernel(X, Y, gamma):
    """RBF kernel with C++ acceleration (avoids 3D intermediate array)."""
    X, Y = _pairwise_inputs(X, Y)
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
    X, Y = _pairwise_inputs(X, Y)
    xn2 = float_array(xn2, 1, "xn2")
    yn2 = float_array(yn2, 1, "yn2")
    if xn2.shape != (X.shape[0],) or yn2.shape != (Y.shape[0],):
        raise ValueError("precomputed norms must match the number of samples")
    cpp = _load_cpp_kernels()
    if cpp is not None:
        return cpp.rbf_kernel_fast(X, Y,
                                   xn2,
                                   yn2,
                                   float(gamma))
    d2 = np.maximum(xn2[:, None] + yn2[None, :] - 2.0 * (X @ Y.T), 0.0)
    return np.exp(-float(gamma) * d2)


def cpp_laplacian_kernel(X, Y, gamma):
    X, Y = _pairwise_inputs(X, Y)
    cpp = _load_cpp_kernels()
    if cpp is not None:
        return cpp.laplacian_kernel(X, Y, float(gamma))
    l1 = np.sum(np.abs(X[:, None, :] - Y[None, :, :]), axis=2)
    return np.exp(-float(gamma) * l1)


def cpp_chi2_kernel(X, Y, gamma):
    X, Y = _pairwise_inputs(X, Y)
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
    X, Y = _pairwise_inputs(X, Y)
    if np.any(X < 0) or np.any(Y < 0):
        raise ValueError("additive_chi2_kernel expects non-negative inputs")
    cpp = _load_cpp_kernels()
    if cpp is not None:
        return cpp.additive_chi2_kernel(X, Y)
    num = 2.0 * X[:, None, :] * Y[None, :, :]
    den = X[:, None, :] + Y[None, :, :] + 1e-12
    return np.sum(num / den, axis=2)


def cpp_hellinger_kernel(X, Y):
    X, Y = _pairwise_inputs(X, Y)
    if np.any(X < 0) or np.any(Y < 0):
        raise ValueError("hellinger_kernel expects non-negative inputs")
    cpp = _load_cpp_kernels()
    if cpp is not None:
        value = cpp.hellinger_kernel(X, Y)
        if np.all(np.isfinite(value)):
            return value
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
