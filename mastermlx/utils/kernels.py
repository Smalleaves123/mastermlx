from __future__ import annotations

import weakref
from typing import Any

import numpy as np

from ..config import get_backend

try:
    from ._kernel_scalar_ops import (
        cosine_kernel as _cy_cosine_kernel,
        linear_kernel as _cy_linear_kernel,
        poly_kernel as _cy_poly_kernel,
        sigmoid_kernel as _cy_sigmoid_kernel,
    )
except ImportError:  # pragma: no cover - fallback when Cython extensions are unavailable
    _cy_cosine_kernel = None
    _cy_linear_kernel = None
    _cy_poly_kernel = None
    _cy_sigmoid_kernel = None


_PAIRWISE_KERNEL_CACHE: dict[tuple[Any, ...], Any] = {}


def _validate_same_shape(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")
    return X, Y


def _validate_nonnegative(name, X, Y):
    if np.any(X < 0) or np.any(Y < 0):
        raise ValueError(f"{name} expects non-negative inputs")


def resolve_gamma(gamma, n_features):
    if gamma is None or gamma == "scale":
        return 1.0 / max(int(n_features), 1)
    return float(gamma)


def _squared_norms(X):
    X = np.asarray(X, dtype=float)
    return np.sum(X * X, axis=1)


def _cache_key(X, Y, kernel, gamma, coef0, degree):
    return (
        id(X),
        id(Y),
        kernel,
        float(gamma),
        float(coef0),
        int(degree),
        X.shape,
        Y.shape,
    )


def _cached_pairwise_kernel(X, Y, kernel, gamma, coef0, degree, compute):
    key = _cache_key(X, Y, kernel, gamma, coef0, degree)
    entry = _PAIRWISE_KERNEL_CACHE.get(key)
    if entry is not None:
        x_ref, y_ref, value = entry
        if x_ref() is X and y_ref() is Y:
            return value
    value = compute()
    _PAIRWISE_KERNEL_CACHE[key] = (weakref.ref(X), weakref.ref(Y), value)
    if len(_PAIRWISE_KERNEL_CACHE) > 8:
        _PAIRWISE_KERNEL_CACHE.pop(next(iter(_PAIRWISE_KERNEL_CACHE)))
    return value


# ============================================================================
#  Individual kernel functions  —  each tries C++ first, falls back to NumPy
# ============================================================================

def linear_kernel(X, Y):
    X, Y = _validate_same_shape(X, Y)
    if get_backend() != "numpy" and _cy_linear_kernel is not None:
        return _cy_linear_kernel(X, Y)
    return X @ Y.T


def cosine_kernel(X, Y):
    X, Y = _validate_same_shape(X, Y)
    if get_backend() != "numpy" and _cy_cosine_kernel is not None:
        return _cy_cosine_kernel(X, Y)
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True).T
    return (X @ Y.T) / (X_norm * Y_norm + 1e-12)


def poly_kernel(X, Y, gamma, coef0, degree):
    if get_backend() != "numpy" and _cy_poly_kernel is not None:
        X, Y = _validate_same_shape(X, Y)
        return _cy_poly_kernel(X, Y, float(gamma), float(coef0), int(degree))
    return (gamma * linear_kernel(X, Y) + coef0) ** degree


def rbf_kernel(X, Y, gamma):
    """RBF (Gaussian) kernel — C++ accelerated, avoids 3D intermediate array."""
    X, Y = _validate_same_shape(X, Y)
    if X is Y:
        return _cached_pairwise_kernel(X, Y, "rbf", float(gamma), 0.0, 0, lambda: _rbf_kernel_impl(X, Y, gamma))
    return _rbf_kernel_impl(X, Y, gamma)


def _rbf_kernel_impl(X, Y, gamma):
    if X is Y or np.array_equal(X, Y):
        from ..accel import cpp_rbf_kernel_fast as _accel_fast
        norms = _squared_norms(X)
        return _accel_fast(X, Y, norms, norms, float(gamma))
    from ..accel import cpp_rbf_kernel as _accel
    return _accel(X, Y, float(gamma))


def laplacian_kernel(X, Y, gamma):
    """Laplacian kernel — C++ accelerated, avoids 3D intermediate array."""
    X, Y = _validate_same_shape(X, Y)
    from ..accel import cpp_laplacian_kernel as _accel
    if X is Y:
        return _cached_pairwise_kernel(X, Y, "laplacian", float(gamma), 0.0, 0, lambda: _accel(X, Y, float(gamma)))
    return _accel(X, Y, float(gamma))


def sigmoid_kernel(X, Y, gamma, coef0):
    if get_backend() != "numpy" and _cy_sigmoid_kernel is not None:
        X, Y = _validate_same_shape(X, Y)
        return _cy_sigmoid_kernel(X, Y, float(gamma), float(coef0))
    return np.tanh(gamma * linear_kernel(X, Y) + coef0)


def chi2_kernel(X, Y, gamma):
    """Chi-squared kernel — C++ accelerated, avoids 3D intermediate array."""
    X, Y = _validate_same_shape(X, Y)
    _validate_nonnegative("chi2_kernel", X, Y)
    from ..accel import cpp_chi2_kernel as _accel
    return _accel(X, Y, gamma)


def additive_chi2_kernel(X, Y):
    """Additive chi-squared kernel — C++ accelerated."""
    X, Y = _validate_same_shape(X, Y)
    _validate_nonnegative("additive_chi2_kernel", X, Y)
    from ..accel import cpp_additive_chi2_kernel as _accel
    return _accel(X, Y)


def hellinger_kernel(X, Y):
    """Hellinger kernel — C++ accelerated."""
    X, Y = _validate_same_shape(X, Y)
    _validate_nonnegative("hellinger_kernel", X, Y)
    from ..accel import cpp_hellinger_kernel as _accel
    return _accel(X, Y)


# ============================================================================
#  Unified dispatcher
# ============================================================================

def pairwise_kernel(X, Y, kernel="rbf", gamma=None, coef0=0.0, degree=3):
    X, Y = _validate_same_shape(X, Y)
    kernel = kernel.lower()
    gamma = resolve_gamma(gamma, X.shape[1])
    if kernel == "linear":
        return linear_kernel(X, Y)
    if kernel == "cosine":
        return cosine_kernel(X, Y)
    if kernel == "poly":
        return poly_kernel(X, Y, gamma, coef0, degree)
    if kernel == "rbf":
        return rbf_kernel(X, Y, gamma)
    if kernel == "laplacian":
        return laplacian_kernel(X, Y, gamma)
    if kernel == "sigmoid":
        return sigmoid_kernel(X, Y, gamma, coef0)
    if kernel == "chi2":
        return chi2_kernel(X, Y, gamma)
    if kernel == "additive_chi2":
        return additive_chi2_kernel(X, Y)
    if kernel == "hellinger":
        return hellinger_kernel(X, Y)
    raise ValueError(
        "kernel must be one of: linear, cosine, poly, rbf, laplacian, "
        "sigmoid, chi2, additive_chi2, hellinger"
    )
