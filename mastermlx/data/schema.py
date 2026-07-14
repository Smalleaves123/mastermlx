"""Feature schema comparison helpers."""

from __future__ import annotations

from .quality import _as_table, _dtype_at


def compare_schema(train, test):
    """Compare feature names, order, widths, and coarse dtypes."""

    train_arr, train_names = _as_table(train)
    test_arr, test_names = _as_table(test)
    train_map = {name: idx for idx, name in enumerate(train_names)}
    test_map = {name: idx for idx, name in enumerate(test_names)}
    common = [name for name in train_names if name in test_map]
    dtype_changes = []
    for name in common:
        train_idx = train_map[name]
        test_idx = test_map[name]
        train_dtype = _dtype_at(train, train_idx, train_arr[:, train_idx])
        test_dtype = _dtype_at(test, test_idx, test_arr[:, test_idx])
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


schema_diff = compare_schema

__all__ = ["compare_schema", "schema_diff"]
