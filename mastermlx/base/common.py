from __future__ import annotations

from ..utils.validation import (
    check_X,
    check_X_y,
    check_feature_count,
    check_is_fitted,
    set_n_features,
)


class BaseAPI:
    """Shared validation helpers for estimators and transformers."""

    def _check_X(self, X, *, dtype=None, allow_1d=False):
        X = check_X(X, dtype=dtype, allow_1d=allow_1d)
        n_features = getattr(self, "n_features_in_", None)
        if n_features is not None:
            check_feature_count(X, n_features)
        return X

    def _set_n_features(self, X):
        set_n_features(self, X)
        return self

    def _check_X_y(self, X, y, *, dtype=None, y_dtype=None):
        X, y = check_X_y(X, y, dtype=dtype, y_dtype=y_dtype)
        n_features = getattr(self, "n_features_in_", None)
        if n_features is not None:
            check_feature_count(X, n_features)
        return X, y

    def _check_fitted(self, attributes=None):
        return check_is_fitted(self, attributes)
