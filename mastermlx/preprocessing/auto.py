"""Automatic tabular feature type detection and preprocessing."""

from __future__ import annotations

import numbers

import numpy as np

from ..base import BaseTransformer
from .column import ColumnTransformer
from .encoders import OneHotEncoder
from .imputers import SimpleImputer
from .pipeline import Pipeline
from .scalers import StandardScaler


def _table(X):
    arr = np.asarray(X)
    if arr.size == 0:
        raise ValueError("X must be non-empty")
    if arr.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {arr.shape}")
    raw_names = getattr(X, "columns", None)
    names = [f"x{idx}" for idx in range(arr.shape[1])] if raw_names is None else list(raw_names)
    if len(names) != arr.shape[1]:
        raise ValueError("X.columns must match the number of columns in X")
    if len(set(names)) != len(names):
        raise ValueError("X.columns must be unique")
    return arr, names


def _missing(value):
    if value is None:
        return True
    try:
        result = np.isnan(value)
        return bool(result) if np.ndim(result) == 0 else False
    except (TypeError, ValueError):
        return False


def _numeric(col, dtype=None):
    if dtype is not None:
        try:
            if np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.bool_):
                return True
            if np.issubdtype(dtype, np.bool_):
                return False
        except TypeError:
            pass
    if np.issubdtype(col.dtype, np.number) and not np.issubdtype(col.dtype, np.bool_):
        return True
    values = [value for value in col if not _missing(value)]
    return bool(values) and all(
        isinstance(value, numbers.Number) and not isinstance(value, (bool, np.bool_))
        for value in values
    )


def _dtype_at(X, idx):
    dtypes = getattr(X, "dtypes", None)
    if dtypes is None:
        return None
    try:
        return dtypes.iloc[idx]
    except AttributeError:
        try:
            return list(dtypes)[idx]
        except (TypeError, IndexError):
            return None


def _resolve(cols, names, n_cols):
    if cols is None or (isinstance(cols, str) and cols == "auto"):
        return []
    if isinstance(cols, slice):
        result = np.arange(n_cols)[cols].tolist()
    else:
        values = [cols] if isinstance(cols, (str, numbers.Integral)) else list(cols)
        result = []
        for value in values:
            if isinstance(value, str):
                if names is None or value not in names:
                    raise ValueError(f"unknown column {value!r}")
                result.append(names.index(value))
            else:
                value = int(value)
                if value < 0 or value >= n_cols:
                    raise ValueError(f"column index out of range [0, {n_cols})")
                result.append(value)
    if len(result) != len(set(result)):
        raise ValueError("column selectors must not contain duplicates")
    return result


def _auto(cols):
    return cols is None or (isinstance(cols, str) and cols == "auto")


class AutoPreprocessor(BaseTransformer):
    """Build a safe numeric/categorical preprocessing pipeline automatically.

    ``num_cols`` and ``cat_cols`` may contain integer positions, column names,
    or slices.  Leaving both unset enables per-column type detection.
    """

    def __init__(
        self,
        num_cols=None,
        cat_cols=None,
        remainder="drop",
        scale=True,
        handle_unknown="ignore",
    ):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.remainder = remainder
        self.scale = scale
        self.handle_unknown = handle_unknown
        self.transformer_ = None
        self.numeric_cols_ = None
        self.categorical_cols_ = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None

    def _select(self, X):
        arr, names = _table(X)
        n_cols = arr.shape[1]
        explicit_num = _resolve(self.num_cols, names, n_cols)
        explicit_cat = _resolve(self.cat_cols, names, n_cols)
        if set(explicit_num) & set(explicit_cat):
            raise ValueError("num_cols and cat_cols must not overlap")

        dtypes = [_dtype_at(X, idx) for idx in range(n_cols)]
        detected_num = [idx for idx in range(n_cols) if _numeric(arr[:, idx], dtypes[idx])]
        detected_cat = [idx for idx in range(n_cols) if idx not in detected_num]
        num = explicit_num if not _auto(self.num_cols) else [
            idx for idx in detected_num if idx not in explicit_cat
        ]
        cat = explicit_cat if not _auto(self.cat_cols) else [
            idx for idx in detected_cat if idx not in explicit_num
        ]
        used = set(num) | set(cat)
        if self.remainder == "passthrough":
            remainder = [idx for idx in range(n_cols) if idx not in used]
        elif self.remainder == "drop":
            remainder = []
        else:
            raise ValueError("remainder must be 'drop' or 'passthrough'")
        return arr, names, num, cat, remainder

    def fit(self, X, y=None):
        arr, names, num, cat, _ = self._select(X)
        if self.handle_unknown not in {"error", "ignore"}:
            raise ValueError("handle_unknown must be 'error' or 'ignore'")
        transformers = []
        if num:
            num_steps = [("impute", SimpleImputer(strategy="median"))]
            if self.scale:
                num_steps.append(("scale", StandardScaler()))
            numeric = num_steps[0][1] if len(num_steps) == 1 else Pipeline(num_steps)
            transformers.append(("num", numeric, num))
        if cat:
            categorical = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("encode", OneHotEncoder(handle_unknown=self.handle_unknown)),
                ]
            )
            transformers.append(("cat", categorical, cat))

        self.transformer_ = ColumnTransformer(transformers, remainder=self.remainder).fit(X, y)
        self._set_n_features(arr)
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.numeric_cols_ = np.asarray([names[idx] if getattr(X, "columns", None) is not None else idx for idx in num], dtype=object)
        self.categorical_cols_ = np.asarray([names[idx] if getattr(X, "columns", None) is not None else idx for idx in cat], dtype=object)
        self.feature_names_out_ = self.get_feature_names_out()
        return self

    def transform(self, X):
        self._check_fitted(["transformer_", "feature_names_in_"])
        incoming = getattr(X, "columns", None)
        if incoming is not None and list(incoming) != self.feature_names_in_.tolist():
            raise ValueError("X columns do not match the fitted schema")
        return self.transformer_.transform(X)

    def get_feature_names_out(self, input_features=None):
        self._check_fitted("transformer_")
        if input_features is None:
            input_features = self.feature_names_in_
        return self.transformer_.get_feature_names_out(input_features)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def make_preprocessor(X=None, **kwargs):
    """Create an :class:`AutoPreprocessor`, fitting it when ``X`` is given."""

    processor = AutoPreprocessor(**kwargs)
    return processor if X is None else processor.fit(X)


__all__ = ["AutoPreprocessor", "make_preprocessor"]
