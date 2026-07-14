from __future__ import annotations

import numpy as np


class NotFittedError(RuntimeError, AttributeError):
    """Raised when an estimator or transformer is used before fitting."""


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


def check_X(X, *, dtype=None, allow_1d=False):
    """Validate a feature matrix and optionally coerce its dtype."""

    X = as_2d(X) if allow_1d else check_2d_array(X)
    return X.astype(dtype) if dtype is not None else X


def check_X_y(X, y, *, dtype=None, y_dtype=None):
    """Validate a feature matrix and target vector together."""

    X = check_X(X, dtype=dtype)
    y = check_1d_array(y)
    if y_dtype is not None:
        y = y.astype(y_dtype)
    return check_same_rows(X, y)


def set_n_features(estimator, X):
    """Record the number of input features seen during fitting."""

    if np.asarray(X).ndim != 2:
        raise ValueError("X must be 2D when recording feature count")
    estimator.n_features_in_ = int(X.shape[1])
    return estimator


def check_feature_count(X, n_features):
    """Ensure a feature matrix matches a fitted estimator's feature count."""

    if int(X.shape[1]) != int(n_features):
        raise ValueError(
            "X has a different number of features than the fitted data "
            f"({X.shape[1]} != {n_features})"
        )
    return X


def check_is_fitted(estimator, attributes=None):
    """Check that fitted attributes exist and are not ``None``."""

    if attributes is None:
        attributes = [
            name
            for name, value in vars(estimator).items()
            if name.endswith("_") and not name.startswith("_") and value is not None
        ]
        if attributes:
            return estimator
        raise NotFittedError(f"{type(estimator).__name__} has not been fit yet")

    if isinstance(attributes, str):
        attributes = [attributes]
    missing = [name for name in attributes if getattr(estimator, name, None) is None]
    if missing:
        names = ", ".join(missing)
        raise NotFittedError(f"{type(estimator).__name__} is missing fitted attributes: {names}")
    return estimator
