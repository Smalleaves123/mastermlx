from __future__ import annotations

import numpy as np


def check_2d_array(X):
    X = np.asarray(X)
    if X.size == 0:
        raise ValueError("Expected a non-empty array")
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")
    return X


def check_1d_array(y, name="y"):
    y = np.asarray(y)
    if y.size == 0:
        raise ValueError(f"Expected {name} to be non-empty")
    if y.ndim != 1:
        raise ValueError(f"Expected {name} to be 1D, got shape {y.shape}")
    return y


def check_same_rows(X, y):
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    return X, y


def as_2d(X):
    X = np.asarray(X)
    if X.size == 0:
        raise ValueError("Expected a non-empty array")
    if X.ndim == 1:
        return X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {X.shape}")
    return X
