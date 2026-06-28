from __future__ import annotations

import numpy as np


class LabelEncoder:
    """Encode labels as contiguous integers."""

    def __init__(self):
        self.classes_ = None

    def fit(self, y):
        y = np.asarray(y)
        if y.ndim != 1 or y.size == 0:
            raise ValueError("y must be a non-empty 1D array")
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        y = np.asarray(y)
        if self.classes_ is None:
            raise RuntimeError("Encoder has not been fit yet")
        index = {label: idx for idx, label in enumerate(self.classes_)}
        try:
            out = np.array([index[item] for item in y], dtype=int)
        except KeyError as exc:
            raise ValueError("y contains unseen labels") from exc
        return int(out[0]) if out.ndim == 1 and out.shape[0] == 1 else out

    def inverse_transform(self, y):
        y = np.asarray(y, dtype=int)
        if self.classes_ is None:
            raise RuntimeError("Encoder has not been fit yet")
        out = self.classes_[y]
        return out.item() if out.ndim == 0 else out

    def fit_transform(self, y):
        return self.fit(y).transform(y)


class OneHotEncoder:
    """One-hot encode categorical columns."""

    def __init__(self, drop=None):
        self.drop = drop
        self.categories_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=object)
        if X.size == 0:
            raise ValueError("Expected a non-empty array")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object)
        if self.categories_ is None:
            raise RuntimeError("Encoder has not been fit yet")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != len(self.categories_):
            raise ValueError("X has a different number of columns than the fitted data")

        parts = []
        for j, cats in enumerate(self.categories_):
            cur = X[:, j]
            if not np.all(np.isin(cur, cats)):
                raise ValueError("X contains unseen categories")
            use_cats = cats[1:] if self.drop == "first" else cats
            block = np.zeros((X.shape[0], len(use_cats)), dtype=float)
            for idx, cat in enumerate(use_cats):
                block[:, idx] = (cur == cat).astype(float)
            parts.append(block)
        if not parts:
            return np.empty((X.shape[0], 0), dtype=float)
        return np.hstack(parts)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class OrdinalEncoder:
    """Encode categorical columns as ordinal integers."""

    def __init__(self):
        self.categories_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=object)
        if X.size == 0:
            raise ValueError("Expected a non-empty array")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object)
        if self.categories_ is None:
            raise RuntimeError("Encoder has not been fit yet")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != len(self.categories_):
            raise ValueError("X has a different number of columns than the fitted data")

        out = np.zeros(X.shape, dtype=float)
        for j, cats in enumerate(self.categories_):
            cur = X[:, j]
            if not np.all(np.isin(cur, cats)):
                raise ValueError("X contains unseen categories")
            index = {cat: idx for idx, cat in enumerate(cats)}
            out[:, j] = [index[item] for item in cur]
        return out

    def inverse_transform(self, X):
        X = np.asarray(X)
        if self.categories_ is None:
            raise RuntimeError("Encoder has not been fit yet")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        out = np.empty(X.shape, dtype=object)
        for j, cats in enumerate(self.categories_):
            idx = X[:, j].astype(int)
            out[:, j] = cats[idx]
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
