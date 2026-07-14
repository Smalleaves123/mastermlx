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
    """Compare feature names, order, widths, and coarse dtypes."""

    train_arr, train_names = _as_table(train)
    test_arr, test_names = _as_table(test)
    train_map = {name: idx for idx, name in enumerate(train_names)}
    test_map = {name: idx for idx, name in enumerate(test_names)}
    common = [name for name in train_names if name in test_map]
    dtype_changes = []
    for name in common:
        train_dtype = _dtype_at(train, train_map[name], train_arr[:, train_map[name]])
        test_dtype = _dtype_at(test, test_map[name], test_arr[:, test_map[name]])
        if train_dtype != test_dtype:
            dtype_changes.append({"name": name, "train": train_dtype, "test": test_dtype})
    return {
        "train_shape": tuple(int(value) for value in train_arr.shape),
        "test_shape": tuple(int(value) for value in test_arr.shape),
        "train_columns": list(train_names),
        "test_columns": list(test_names),
        "missing_columns": [name for name in train_names if name not in test_map],
        "extra_columns": [name for name in test_names if name not in train_map],
        "order_match": train_names == test_names,
        "dtype_changes": dtype_changes,
    }


def drift_report(train, test, *, bins=10):
    """Measure simple per-column distribution changes between two tables."""

    if int(bins) < 2:
        raise ValueError("bins must be at least 2")
    bins = int(bins)
    train_arr, train_names = _as_table(train)
    test_arr, test_names = _as_table(test)
    test_map = {name: idx for idx, name in enumerate(test_names)}
    common = [name for name in train_names if name in test_map]
    columns = []

    for idx, name in enumerate(train_names):
        if name not in test_map:
            continue
        test_idx = test_map[name]
        train_col = train_arr[:, idx]
        test_col = test_arr[:, test_idx]
        train_missing = _missing_mask(train_col)
        test_missing = _missing_mask(test_col)
        item = {
            "name": name,
            "train_missing_rate": float(train_missing.mean()),
            "test_missing_rate": float(test_missing.mean()),
            "missing_rate_diff": float(test_missing.mean() - train_missing.mean()),
        }
        if _is_numeric(train_col, train_missing) and _is_numeric(test_col, test_missing):
            a = np.asarray(train_col[~train_missing], dtype=float)
            b = np.asarray(test_col[~test_missing], dtype=float)
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]
            if a.size and b.size:
                edges = np.histogram_bin_edges(a, bins=bins)
                if np.all(edges == edges[0]):
                    edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
                train_hist, _ = np.histogram(a, bins=edges)
                test_hist, _ = np.histogram(b, bins=edges)
                train_prob = (train_hist + 1.0) / (train_hist.sum() + len(train_hist))
                test_prob = (test_hist + 1.0) / (test_hist.sum() + len(test_hist))
                item.update(
                    kind="numeric",
                    train_mean=float(np.mean(a)),
                    test_mean=float(np.mean(b)),
                    mean_diff=float(np.mean(b) - np.mean(a)),
                    train_std=float(np.std(a)),
                    test_std=float(np.std(b)),
                    psi=float(np.sum((test_prob - train_prob) * np.log(test_prob / train_prob))),
                )
            else:
                item.update(kind="numeric", train_mean=np.nan, test_mean=np.nan, mean_diff=np.nan, psi=np.nan)
        else:
            train_counts = dict((_key(value), count) for value, count in _counts(train_col[~train_missing]))
            test_counts = dict((_key(value), count) for value, count in _counts(test_col[~test_missing]))
            keys = set(train_counts) | set(test_counts)
            train_total = max(1, sum(train_counts.values()))
            test_total = max(1, sum(test_counts.values()))
            tvd = 0.5 * sum(
                abs(train_counts.get(key, 0) / train_total - test_counts.get(key, 0) / test_total)
                for key in keys
            )
            item.update(
                kind="categorical",
                train_unique_count=len(train_counts),
                test_unique_count=len(test_counts),
                unseen_rate=float(
                    sum(1 for value in test_col[~test_missing] if _key(value) not in train_counts)
                    / max(1, (~test_missing).sum())
                ),
                tvd=float(tvd),
            )
        columns.append(item)
    return {"schema": compare_schema(train, test), "columns": columns, "common_columns": common}


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
