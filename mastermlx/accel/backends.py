from __future__ import annotations

import importlib

import numpy as np

from ..config import get_backend


def _numpy_pairwise_squared_euclidean(X, Y):
    diff = X[:, None, :] - Y[None, :, :]
    return np.sum(diff * diff, axis=2)


def _numpy_pairwise_distances(X, Y):
    return np.sqrt(np.maximum(_numpy_pairwise_squared_euclidean(X, Y), 0.0))


def _numpy_pairwise_manhattan(X, Y):
    diff = np.abs(X[:, None, :] - Y[None, :, :])
    return np.sum(diff, axis=2)


def _load_cython_backend():
    try:
        return importlib.import_module("mastermlx.accel._distance_ops")
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


def _load_cpp_backend():
    try:
        return importlib.import_module("mastermlx.accel._distance_cpp")
    except ImportError:
        return None


def pairwise_squared_euclidean(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    # Try C++ first, then Cython, then NumPy
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


active_backend = get_active_backend
pairwise_sq_euclid = pairwise_squared_euclidean


def _load_cython_tree():
    try:
        return importlib.import_module("mastermlx.accel._tree_ops")
    except ImportError:
        return None


def best_split_classifier(X, y, min_samples_leaf):
    mod = _load_cython_tree()
    if mod is not None:
        return mod._best_split_classifier(X, y, min_samples_leaf)
    # Fallback handled in the decision tree itself
    return None


def best_split_regressor(X, y, min_samples_leaf):
    mod = _load_cython_tree()
    if mod is not None:
        return mod._best_split_regressor(X, y, min_samples_leaf)
    return None
pairwise_manhattan = pairwise_manhattan_distances
pairwise_dist = pairwise_distances
