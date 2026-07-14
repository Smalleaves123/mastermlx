"""Data quality, schema, and lightweight drift checks."""

from __future__ import annotations

import numbers
from copy import deepcopy

import numpy as np


def _as_table(X):
    arr = np.asarray(X)
    if arr.size == 0:
        raise ValueError("X must be non-empty")
    if arr.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {arr.shape}")

    raw_names = getattr(X, "columns", None)
    if raw_names is None:
        names = [f"x{idx}" for idx in range(arr.shape[1])]
    else:
        names = list(raw_names)
        if len(names) != arr.shape[1]:
            raise ValueError("X.columns must match the number of columns in X")
    return arr, names


def _missing(value):
    if value is None:
        return True
    try:
        result = np.isnan(value)
        return bool(result) if np.ndim(result) == 0 else False
    except (TypeError, ValueError):
        return False


def _missing_mask(col):
    if np.issubdtype(col.dtype, np.inexact):
        return np.isnan(col)
    return np.asarray([_missing(value) for value in col], dtype=bool)


def _key(value):
    if isinstance(value, np.generic):
        value = value.item()
    try:
        hash(value)
        return (type(value).__name__, value)
    except TypeError:
        return (type(value).__name__, repr(value))


def _counts(values):
    result = {}
    display = {}
    for value in values:
        key = _key(value)
        result[key] = result.get(key, 0) + 1
        display[key] = value.item() if isinstance(value, np.generic) else value
    return [(display[key], count) for key, count in result.items()]


def _is_numeric(col, mask):
    if np.issubdtype(col.dtype, np.number) and not np.issubdtype(col.dtype, np.bool_):
        return True
    values = col[~mask]
    return bool(values.size) and all(
        isinstance(value, numbers.Number) and not isinstance(value, (bool, np.bool_))
        for value in values
    )


def _dtype(col):
    return str(col.dtype)


def _dtype_at(X, idx, col):
    dtypes = getattr(X, "dtypes", None)
    if dtypes is not None:
        try:
            return str(dtypes.iloc[idx])
        except AttributeError:
            try:
                return str(list(dtypes)[idx])
            except (TypeError, IndexError):
                pass
    return _dtype(col)


def _numeric_stats(values, outlier=1.5):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    result = {
        "mean": np.nan,
        "std": np.nan,
        "min": np.nan,
        "max": np.nan,
        "median": np.nan,
        "q25": np.nan,
        "q75": np.nan,
        "non_finite_count": int(values.size - finite.size),
        "outlier_count": 0,
    }
    if finite.size == 0:
        return result

    q25, q75 = np.percentile(finite, [25.0, 75.0])
    result.update(
        mean=float(np.mean(finite)),
        std=float(np.std(finite)),
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        median=float(np.median(finite)),
        q25=float(q25),
        q75=float(q75),
    )
    spread = q75 - q25
    if spread == 0.0:
        result["outlier_count"] = int(np.sum(finite != q25))
    else:
        low = q25 - float(outlier) * spread
        high = q75 + float(outlier) * spread
        result["outlier_count"] = int(np.sum((finite < low) | (finite > high)))
    return result


def _target_report(y):
    y = np.asarray(y)
    if y.size == 0 or y.ndim != 1:
        raise ValueError("y must be a non-empty 1D array")
    mask = _missing_mask(y)
    counts = _counts(y[~mask])
    return {
        "dtype": str(y.dtype),
        "missing_count": int(mask.sum()),
        "missing_rate": float(mask.mean()),
        "unique_count": len(counts),
        "value_counts": [
            {"value": value, "count": int(count), "rate": float(count / y.size)}
            for value, count in sorted(counts, key=lambda item: item[1], reverse=True)
        ],
    }


def quality_report(X, y=None, *, low_freq=0.01, outlier=1.5, top=5):
    """Build a compact quality report for a tabular feature matrix.

    The function accepts NumPy-like data and table objects exposing ``columns``
    and optionally ``dtypes``.  Numeric columns include robust summary values
    and IQR-based outlier counts; categorical columns include frequent values
    and low-frequency categories.
    """

    if not 0.0 <= low_freq <= 1.0:
        raise ValueError("low_freq must be between 0 and 1")
    if outlier < 0.0:
        raise ValueError("outlier must be non-negative")
    if int(top) < 0:
        raise ValueError("top must be non-negative")
    top = int(top)

    arr, names = _as_table(X)
    n_rows, n_cols = arr.shape
    columns = []
    numeric_columns = []
    categorical_columns = []
    constant_columns = []
    total_missing = 0

    for idx, name in enumerate(names):
        col = arr[:, idx]
        mask = _missing_mask(col)
        values = col[~mask]
        is_num = _is_numeric(col, mask)
        counts = _counts(values)
        unique_count = len(counts)
        missing_count = int(mask.sum())
        total_missing += missing_count

        item = {
            "name": name,
            "index": idx,
            "dtype": _dtype(col),
            "kind": "numeric" if is_num else "categorical",
            "missing_count": missing_count,
            "missing_rate": float(missing_count / n_rows),
            "unique_count": unique_count,
            "constant": unique_count <= 1,
        }
        if is_num:
            numeric_columns.append(name)
            item.update(_numeric_stats(values, outlier=outlier))
        else:
            categorical_columns.append(name)
            value_counts = sorted(counts, key=lambda pair: (-pair[1], repr(pair[0])))
            item["value_counts"] = [
                {"value": value, "count": int(count), "rate": float(count / n_rows)}
                for value, count in value_counts[:top]
            ]
            item["low_freq_values"] = [
                value for value, count in counts if count / n_rows < low_freq
            ]
        if item["constant"]:
            constant_columns.append(name)
        columns.append(item)

    row_keys = [tuple(_key(value) if not _missing(value) else ("missing",) for value in row) for row in arr]
    duplicate_rows = n_rows - len(set(row_keys))
    return {
        "n_rows": int(n_rows),
        "n_columns": int(n_cols),
        "columns": columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "constant_columns": constant_columns,
        "duplicate_rows": int(duplicate_rows),
        "missing": {
            "count": int(total_missing),
            "rate": float(total_missing / (n_rows * n_cols)),
        },
        "target": None if y is None else _target_report(y),
    }


def compare_schema(train, test):
    """Compatibility wrapper for :func:`mastermlx.data.schema.compare_schema`."""

    from .schema import compare_schema as _compare_schema

    return _compare_schema(train, test)


def drift_report(train, test, *, bins=10):
    """Compatibility wrapper for :func:`mastermlx.data.drift.drift_report`."""

    from .drift import drift_report as _drift_report

    return _drift_report(train, test, bins=bins)


class DataQualityReport:
    """Reusable wrapper around :func:`quality_report`."""

    def __init__(self, *, low_freq=0.01, outlier=1.5, top=5):
        self.low_freq = low_freq
        self.outlier = outlier
        self.top = top
        self.report_ = None
        self.reference_ = None

    def fit(self, X, y=None):
        self.reference_ = deepcopy(X)
        self.report_ = quality_report(
            X, y, low_freq=self.low_freq, outlier=self.outlier, top=self.top
        )
        return self

    def report(self):
        if self.report_ is None:
            raise RuntimeError("DataQualityReport has not been fit yet")
        return self.report_

    def compare(self, X, *, drift=False):
        if self.report_ is None:
            raise RuntimeError("DataQualityReport has not been fit yet")
        if drift:
            return drift_report(self.reference_, X)
        current = quality_report(X, low_freq=self.low_freq, outlier=self.outlier, top=self.top)
        return current


data_quality = quality_report
schema_diff = compare_schema
data_drift = drift_report

__all__ = [
    "DataQualityReport",
    "compare_schema",
    "data_drift",
    "data_quality",
    "drift_report",
    "quality_report",
    "schema_diff",
]
