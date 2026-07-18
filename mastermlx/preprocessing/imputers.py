from __future__ import annotations

import numpy as np
from typing import Any

from ..base import BaseTransformer
from ..utils.validation import check_2d_array


def _missing(value):
    if value is None:
        return True
    try:
        result = np.isnan(value)
        return bool(result) if np.ndim(result) == 0 else False
    except (TypeError, ValueError):
        return False


def _mask(X):
    if np.issubdtype(X.dtype, np.inexact):
        return np.isnan(X)
    return np.asarray([[_missing(value) for value in row] for row in X], dtype=bool)


def _count(values):
    counts: list[tuple[Any, int]] = []
    for value in values:
        found = False
        for idx, (item, count) in enumerate(counts):
            try:
                same = bool(value == item)
            except (TypeError, ValueError):
                same = repr(value) == repr(item)
            if same:
                counts[idx] = (item, count + 1)
                found = True
                break
        if not found:
            counts.append((value, 1))
    return counts


class SimpleImputer(BaseTransformer):
    """Fill missing values with a per-column statistic."""

    def __init__(self, strategy="mean", fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_: np.ndarray | None = None
        self._numeric: bool | None = None

    def fit(self, X, y=None):
        X = check_2d_array(X)
        if self.strategy not in {"mean", "median", "most_frequent", "constant"}:
            raise ValueError("strategy must be one of: mean, median, most_frequent, constant")

        mask = _mask(X)
        values = X[~mask]
        raw_numeric = np.issubdtype(X.dtype, np.number) and not np.issubdtype(X.dtype, np.bool_)
        if not raw_numeric and values.size:
            raw_numeric = all(
                isinstance(value, (int, float, complex, np.number))
                and not isinstance(value, (bool, np.bool_))
                for value in values
            )
        self._numeric = self.strategy in {"mean", "median"} or (
            raw_numeric
            and self.strategy in {"most_frequent", "constant"}
            and (self.fill_value is None or isinstance(self.fill_value, (int, float, complex, np.number)))
        )
        if self._numeric:
            X = X.astype(float)
            mask = np.isnan(X)
        stats = []
        for j in range(X.shape[1]):
            col = X[:, j]
            valid = col[~mask[:, j]]
            if self.strategy == "mean":
                value = float(np.mean(valid)) if valid.size else 0.0
            elif self.strategy == "median":
                value = float(np.median(valid)) if valid.size else 0.0
            elif self.strategy == "most_frequent":
                if valid.size == 0:
                    value = 0.0 if self.fill_value is None else self.fill_value
                else:
                    value = max(_count(valid), key=lambda item: item[1])[0]
            else:
                value = 0.0 if self.fill_value is None else self.fill_value
            stats.append(value)
        self.statistics_ = np.asarray(stats, dtype=float if self._numeric else object)
        self._set_n_features(X)
        return self

    def transform(self, X):
        self._check_fitted(["statistics_", "_numeric"])
        statistics = self.statistics_
        numeric = self._numeric
        if statistics is None or numeric is None:
            raise RuntimeError("Imputer has not been fit yet")
        X = check_2d_array(X)
        X = X.astype(float if numeric else object)
        self._check_X(X)
        out = X.copy()
        mask = _mask(out)
        for j in range(out.shape[1]):
            out[mask[:, j], j] = statistics[j]
        return out
