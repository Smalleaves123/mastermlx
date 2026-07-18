from __future__ import annotations

import numpy as np
from typing import Any

from ..base import BaseTransformer


def _unique(values):
    try:
        return np.unique(values)
    except TypeError:
        result: list[Any] = []
        for value in values:
            if not any(_same(value, item) for item in result):
                result.append(value)
        return np.asarray(result, dtype=object)


def _same(left, right):
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return repr(left) == repr(right)


def _find(value, values):
    for idx, item in enumerate(values):
        if _same(value, item):
            return idx
    return None


class LabelEncoder:
    """Encode labels as contiguous integers."""

    def __init__(self):
        self.classes_: np.ndarray | None = None

    def fit(self, y):
        y = np.asarray(y)
        if y.ndim != 1 or y.size == 0:
            raise ValueError("y must be a non-empty 1D array")
        self.classes_ = _unique(y)
        return self

    def transform(self, y):
        y = np.asarray(y)
        if self.classes_ is None:
            raise RuntimeError("Encoder has not been fit yet")
        classes = self.classes_
        if classes is None:
            raise RuntimeError("Encoder has not been fit yet")
        index = {label: idx for idx, label in enumerate(classes)}
        try:
            out = np.array([index[item] for item in y], dtype=int)
        except KeyError as exc:
            raise ValueError("y contains unseen labels") from exc
        return int(out[0]) if out.ndim == 1 and out.shape[0] == 1 else out

    def inverse_transform(self, y):
        y = np.asarray(y, dtype=int)
        if self.classes_ is None:
            raise RuntimeError("Encoder has not been fit yet")
        classes = self.classes_
        if classes is None:
            raise RuntimeError("Encoder has not been fit yet")
        out = classes[y]
        return out.item() if out.ndim == 0 else out

    def fit_transform(self, y):
        return self.fit(y).transform(y)


class OneHotEncoder(BaseTransformer):
    """One-hot encode categorical columns."""

    def __init__(self, drop=None, handle_unknown="error"):
        self.drop = drop
        self.handle_unknown = handle_unknown
        self.categories_: list[np.ndarray] | None = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=object)
        if X.size == 0:
            raise ValueError("Expected a non-empty array")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        if self.drop not in {None, "first"}:
            raise ValueError("drop must be None or 'first'")
        if self.handle_unknown not in {"error", "ignore"}:
            raise ValueError("handle_unknown must be 'error' or 'ignore'")
        self.categories_ = [_unique(X[:, j]) for j in range(X.shape[1])]
        self._set_n_features(X)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object)
        if self.categories_ is None:
            raise RuntimeError("Encoder has not been fit yet")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        categories = self.categories_
        if categories is None:
            raise RuntimeError("Encoder has not been fit yet")
        if X.shape[1] != len(categories):
            raise ValueError("X has a different number of columns than the fitted data")

        parts = []
        for j, cats in enumerate(categories):
            cur = X[:, j]
            known = np.asarray([_find(value, cats) is not None for value in cur])
            if not np.all(known) and self.handle_unknown == "error":
                raise ValueError("X contains unseen categories")
            use_cats = list(cats[1:]) if self.drop == "first" else list(cats)
            block = np.zeros((X.shape[0], len(use_cats)), dtype=float)
            for idx, cat in enumerate(use_cats):
                block[:, idx] = np.asarray([_same(value, cat) for value in cur], dtype=float)
            parts.append(block)
        if not parts:
            return np.empty((X.shape[0], 0), dtype=float)
        return np.hstack(parts)

    def get_feature_names_out(self, input_features=None):
        self._check_fitted("categories_")
        categories = self.categories_
        if categories is None:
            raise RuntimeError("Encoder has not been fit yet")
        if input_features is None:
            input_features = [f"x{idx}" for idx in range(len(categories))]
        input_features = np.asarray(input_features, dtype=object).ravel()
        if input_features.size != len(categories):
            raise ValueError("input_features must match the fitted number of columns")
        names: list[str] = []
        for feature, cats in zip(input_features, categories):
            use_cats = cats[1:] if self.drop == "first" else cats
            names.extend(f"{feature}_{category}" for category in use_cats)
        return np.asarray(names, dtype=object)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class OrdinalEncoder(BaseTransformer):
    """Encode categorical columns as ordinal integers."""

    def __init__(self, handle_unknown="error", unknown_value=-1):
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.categories_: list[np.ndarray] | None = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=object)
        if X.size == 0:
            raise ValueError("Expected a non-empty array")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        if self.handle_unknown not in {"error", "use_encoded_value"}:
            raise ValueError("handle_unknown must be 'error' or 'use_encoded_value'")
        self.categories_ = [_unique(X[:, j]) for j in range(X.shape[1])]
        self._set_n_features(X)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object)
        if self.categories_ is None:
            raise RuntimeError("Encoder has not been fit yet")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        categories = self.categories_
        if categories is None:
            raise RuntimeError("Encoder has not been fit yet")
        if X.shape[1] != len(categories):
            raise ValueError("X has a different number of columns than the fitted data")

        out = np.zeros(X.shape, dtype=float)
        for j, cats in enumerate(categories):
            cur = X[:, j]
            if self.handle_unknown == "error" and any(_find(value, cats) is None for value in cur):
                raise ValueError("X contains unseen categories")
            out[:, j] = [
                self.unknown_value if (idx := _find(item, cats)) is None else idx
                for item in cur
            ]
        return out

    def inverse_transform(self, X):
        X = np.asarray(X)
        categories = self.categories_
        if categories is None:
            raise RuntimeError("Encoder has not been fit yet")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        out = np.empty(X.shape, dtype=object)
        for j, cats in enumerate(categories):
            idx = X[:, j].astype(int)
            if np.any(idx < 0) or np.any(idx >= len(cats)):
                raise ValueError("X contains an invalid encoded value")
            out[:, j] = cats[idx]
        return out

    def get_feature_names_out(self, input_features=None):
        self._check_fitted("categories_")
        categories = self.categories_
        if categories is None:
            raise RuntimeError("Encoder has not been fit yet")
        if input_features is None:
            input_features = [f"x{idx}" for idx in range(len(categories))]
        input_features = np.asarray(input_features, dtype=object).ravel()
        if input_features.size != len(categories):
            raise ValueError("input_features must match the fitted number of columns")
        return input_features.astype(object)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
