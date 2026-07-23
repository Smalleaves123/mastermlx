from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike

from ..utils.validation import (
    check_X,
    check_X_y,
    check_feature_count,
    check_is_fitted,
    set_n_features,
)


API_T = TypeVar("API_T", bound="BaseAPI")


class BaseAPI:
    """Shared validation helpers for estimators and transformers."""

    def _check_X(
        self,
        X: ArrayLike,
        *,
        dtype: Any | None = None,
        allow_1d: bool = False,
    ) -> np.ndarray:
        X = check_X(X, dtype=dtype, allow_1d=allow_1d)
        n_features = getattr(self, "n_features_in_", None)
        if n_features is not None:
            check_feature_count(X, n_features)
        return X

    def _set_n_features(self: API_T, X: ArrayLike) -> API_T:
        set_n_features(self, X)
        return self

    def _check_X_y(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        dtype: Any | None = None,
        y_dtype: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        X, y = check_X_y(X, y, dtype=dtype, y_dtype=y_dtype)
        n_features = getattr(self, "n_features_in_", None)
        if n_features is not None:
            check_feature_count(X, n_features)
        return X, y

    def _check_fitted(self, attributes: str | list[str] | None = None) -> "BaseAPI":
        return check_is_fitted(self, attributes)
