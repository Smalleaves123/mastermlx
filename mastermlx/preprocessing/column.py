from __future__ import annotations

import numpy as np
from typing import Any

from ..base import BaseTransformer
from ..utils.estimator import clone
from ..utils.validation import check_2d_array


class ColumnTransformer(BaseTransformer):
    """Apply cloned transformers to different column subsets."""

    def __init__(self, transformers, remainder="drop"):
        self.transformers = list(transformers)
        self.remainder = remainder
        self._columns = []
        self._names = []
        self._remainder_idx: list[int] | None = None
        self.transformers_: list[tuple[str, Any, np.ndarray]] | None = None
        self.feature_names_in_: np.ndarray | None = None
        self.output_dims_: list[int] | None = None

    @staticmethod
    def _resolve_cols(X, cols, n_cols):
        if cols is None or (isinstance(cols, (list, np.ndarray)) and len(cols) == 0):
            raise ValueError("columns must not be empty")
        if isinstance(cols, slice):
            return np.arange(n_cols)[cols]
        if isinstance(cols, str):
            cols = [cols]
        if isinstance(cols, (list, tuple, np.ndarray)) and any(isinstance(item, str) for item in cols):
            names = getattr(X, "columns", None)
            if names is None:
                raise TypeError("string columns require input with a 'columns' attribute")
            index = {name: idx for idx, name in enumerate(names)}
            try:
                cols = [index[item] for item in cols]
            except KeyError as exc:
                raise ValueError(f"unknown column {exc.args[0]!r}") from None
        cols = np.atleast_1d(np.asarray(cols, dtype=int)).ravel()
        if cols.size == 0:
            raise ValueError("columns must not be empty")
        if np.any(cols < 0) or np.any(cols >= n_cols):
            raise ValueError(f"columns out of range [0, {n_cols})")
        return cols

    @staticmethod
    def _as_block(value, n_rows):
        value = np.asarray(value)
        if value.ndim == 1:
            value = value.reshape(n_rows, 1)
        if value.ndim != 2 or value.shape[0] != n_rows:
            raise ValueError("transformers must return a 2D array with one row per sample")
        return value

    def fit(self, X, y=None):
        raw = X
        X = check_2d_array(X)
        n_cols = X.shape[1]
        if n_cols == 0:
            raise ValueError("X must have at least one column")
        self._set_n_features(X)
        self.n_features_in_ = n_cols
        names = getattr(raw, "columns", None)
        self.feature_names_in_ = None if names is None else np.asarray(list(names), dtype=object)

        used = np.zeros(n_cols, dtype=bool)
        self._columns = []
        self._names = []
        self.transformers_ = []
        self.output_dims_ = []
        for name, trans, cols in self.transformers:
            cols = self._resolve_cols(raw, cols, n_cols)
            if np.any(used[cols]):
                raise ValueError(f"Columns for '{name}' overlap with another transformer")
            used[cols] = True
            obj = clone(trans)
            obj.fit(X[:, cols], y)
            block = self._as_block(obj.transform(X[:, cols]), X.shape[0])
            self._columns.append(cols)
            self._names.append(name)
            self.transformers_.append((name, obj, cols))
            self.output_dims_.append(block.shape[1])

        self._remainder_idx = []
        if self.remainder == "passthrough":
            self._remainder_idx = np.flatnonzero(~used).tolist()
        elif self.remainder != "drop":
            raise ValueError("remainder must be 'drop' or 'passthrough'")
        return self

    def transform(self, X):
        self._check_fitted("transformers_")
        transformers = self.transformers_
        remainder_idx = self._remainder_idx
        if transformers is None or remainder_idx is None:
            raise RuntimeError("ColumnTransformer has not been fit yet")
        incoming = getattr(X, "columns", None)
        feature_names = self.feature_names_in_
        if feature_names is not None and incoming is not None:
            if list(incoming) != feature_names.tolist():
                raise ValueError("X columns do not match the fitted schema")
        X = self._check_X(X)
        parts = [self._as_block(trans.transform(X[:, cols]), X.shape[0]) for _, trans, cols in transformers]
        if remainder_idx:
            parts.append(X[:, remainder_idx])
        if not parts:
            return np.zeros((X.shape[0], 0), dtype=float)
        return np.column_stack(parts) if len(parts) > 1 else parts[0]

    def get_feature_names_out(self, input_features=None):
        """Return stable names for transformed columns."""

        self._check_fitted(["transformers_", "output_dims_"])
        transformers = self.transformers_
        output_dims = self.output_dims_
        if transformers is None or output_dims is None:
            raise RuntimeError("ColumnTransformer has not been fit yet")
        if input_features is None:
            input_features = self.feature_names_in_
        if input_features is None:
            input_features = np.asarray([f"x{idx}" for idx in range(self.n_features_in_)], dtype=object)
        input_features = np.asarray(input_features, dtype=object).ravel()
        if self.n_features_in_ is None:
            raise RuntimeError("ColumnTransformer has not recorded its input feature count")
        if input_features.size != self.n_features_in_:
            raise ValueError("input_features must match the fitted number of columns")

        names = []
        for (name, trans, cols), width in zip(transformers, output_dims):
            custom = None
            if hasattr(trans, "get_feature_names_out"):
                try:
                    custom = np.asarray(trans.get_feature_names_out(input_features[cols]), dtype=object).ravel()
                except TypeError:
                    custom = np.asarray(trans.get_feature_names_out(), dtype=object).ravel()
            if custom is not None and custom.size == width:
                names.extend([f"{name}__{item}" for item in custom])
            elif width == len(cols):
                names.extend([f"{name}__{input_features[col]}" for col in cols])
            else:
                names.extend([f"{name}__x{idx}" for idx in range(width)])

        if self.remainder == "passthrough" and self._remainder_idx:
            names.extend([str(input_features[idx]) for idx in self._remainder_idx])
        return np.asarray(names, dtype=object)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


__all__ = ["ColumnTransformer"]
