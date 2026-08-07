from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

try:
    from scipy.sparse import issparse as _scipy_issparse
except ImportError:  # pragma: no cover - SciPy is an optional runtime dependency
    _scipy_issparse = None


class NotFittedError(RuntimeError, AttributeError):
    """Raised when an estimator or transformer is used before fitting."""


def _is_sparse(X: Any) -> bool:
    return _scipy_issparse is not None and bool(_scipy_issparse(X))


def check_2d_array(X: ArrayLike):
    if _is_sparse(X):
        if len(X.shape) != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"Expected a non-empty 2D array, got shape {X.shape}")
        return X
    X = np.asarray(X)
    if X.size == 0:
        raise ValueError("Expected a non-empty array")
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")
    return X


def check_1d_array(y: ArrayLike | None, name: str = "y") -> np.ndarray:
    if y is None:
        raise ValueError(f"Expected {name} to be non-empty")
    y = np.asarray(y)
    if y.size == 0:
        raise ValueError(f"Expected {name} to be non-empty")
    if y.ndim != 1:
        raise ValueError(f"Expected {name} to be 1D, got shape {y.shape}")
    return y


def check_y(
    y: ArrayLike | None,
    *,
    allow_2d: bool = False,
    dtype: Any | None = None,
    ensure_all_finite: bool = False,
    name: str = "y",
) -> np.ndarray:
    """Validate a target vector or a multi-output target matrix."""

    if y is None:
        raise ValueError(f"Expected {name} to be non-empty")
    y = np.asarray(y, dtype=dtype)
    valid_ndim = {1, 2} if allow_2d else {1}
    if y.size == 0:
        raise ValueError(f"Expected {name} to be non-empty")
    if y.ndim not in valid_ndim:
        expected = "1D or 2D" if allow_2d else "1D"
        raise ValueError(f"Expected {name} to be {expected}, got shape {y.shape}")
    if ensure_all_finite:
        try:
            finite = np.isfinite(y).all()
        except TypeError as exc:
            raise ValueError(f"{name} must contain only finite numeric values") from exc
        if not finite:
            raise ValueError(f"{name} must contain only finite values")
    return y


def check_same_rows(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    return X, y


def as_2d(X: ArrayLike) -> np.ndarray:
    X = np.asarray(X)
    if X.size == 0:
        raise ValueError("Expected a non-empty array")
    if X.ndim == 1:
        return X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {X.shape}")
    return X


def check_X(
    X: ArrayLike,
    *,
    dtype: Any | None = None,
    allow_1d: bool = False,
    ensure_all_finite: bool = False,
):
    """Validate a feature matrix and optionally coerce its dtype."""

    X = as_2d(X) if allow_1d else check_2d_array(X)
    if dtype is not None:
        X = X.astype(dtype)
    if ensure_all_finite:
        values = X.data if _is_sparse(X) else X
        try:
            finite = np.isfinite(values).all()
        except TypeError as exc:
            raise ValueError("X must contain only finite numeric values") from exc
        if not finite:
            raise ValueError("X must contain only finite values")
    return X


def check_X_y(
    X: ArrayLike,
    y: ArrayLike,
    *,
    dtype: Any | None = None,
    y_dtype: Any | None = None,
    ensure_all_finite: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a feature matrix and target vector together."""

    X = check_X(X, dtype=dtype, ensure_all_finite=ensure_all_finite)
    y = check_1d_array(y)
    if y_dtype is not None:
        y = y.astype(y_dtype)
    if ensure_all_finite:
        try:
            finite = np.isfinite(y).all()
        except TypeError as exc:
            raise ValueError("y must contain only finite numeric values") from exc
        if not finite:
            raise ValueError("y must contain only finite values")
    return check_same_rows(X, y)


def check_sample_weight(
    sample_weight: ArrayLike | None,
    n_samples: int,
) -> np.ndarray:
    """Validate per-sample weights and return a floating-point vector."""

    if sample_weight is None:
        return np.ones(int(n_samples), dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if weights.ndim != 1 or weights.shape[0] != int(n_samples):
        raise ValueError("sample_weight must be 1D and have one value per sample")
    if not np.isfinite(weights).all():
        raise ValueError("sample_weight must contain only finite values")
    if np.any(weights < 0.0):
        raise ValueError("sample_weight must be non-negative")
    if not np.any(weights > 0.0):
        raise ValueError("sample_weight must contain at least one positive value")
    return weights


def to_dense(X):
    """Return a NumPy view/copy for algorithms without sparse kernels."""

    return X.toarray() if _is_sparse(X) else np.asarray(X)


def set_n_features(estimator: Any, X: ArrayLike) -> Any:
    """Record the number of input features seen during fitting."""

    X_array = np.asarray(X)
    if X_array.ndim != 2:
        raise ValueError("X must be 2D when recording feature count")
    estimator.n_features_in_ = int(X_array.shape[1])
    return estimator


def check_feature_count(X: np.ndarray, n_features: int) -> np.ndarray:
    """Ensure a feature matrix matches a fitted estimator's feature count."""

    if int(X.shape[1]) != int(n_features):
        raise ValueError(
            "X has a different number of features than the fitted data "
            f"({X.shape[1]} != {n_features})"
        )
    return X


def check_is_fitted(
    estimator: Any,
    attributes: str | list[str] | None = None,
) -> Any:
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
