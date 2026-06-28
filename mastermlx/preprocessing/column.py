from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils.validation import check_2d_array


class ColumnTransformer(BaseTransformer):
    """Apply different transformers to different column subsets."""

    def __init__(self, transformers, remainder="drop"):
        self.transformers = list(transformers)
        self.remainder = remainder
        self._columns = []
        self._names = []
        self._remainder_idx = None

    def fit(self, X, y=None):
        X = check_2d_array(X)
        n_cols = X.shape[1]
        if n_cols == 0:
            raise ValueError("X must have at least one column")
        used = np.zeros(n_cols, dtype=bool)

        self._columns = []
        self._names = []
        for name, trans, cols in self.transformers:
            if cols is None or (isinstance(cols, (list, np.ndarray)) and len(cols) == 0):
                raise ValueError(f"Columns for '{name}' must not be empty")
            if isinstance(cols, slice):
                cols = np.arange(n_cols)[cols]
            cols = np.atleast_1d(np.asarray(cols, dtype=int)).ravel()
            if cols.size == 0:
                raise ValueError(f"Columns for '{name}' resolved to empty")
            if np.any(cols < 0) or np.any(cols >= n_cols):
                raise ValueError(f"Columns for '{name}' out of range [0, {n_cols})")
            if np.any(used[cols]):
                raise ValueError(f"Columns for '{name}' overlap with another transformer")
            used[cols] = True
            self._columns.append(cols)
            self._names.append(name)
            trans.fit(X[:, cols], y)

        if self.remainder == "passthrough":
            self._remainder_idx = np.flatnonzero(~used).tolist()
        elif self.remainder != "drop":
            raise ValueError("remainder must be 'drop' or 'passthrough'")
        return self

    def transform(self, X):
        X = check_2d_array(X)
        parts = []
        for cols, trans in zip(self._columns, self._names):
            trans_obj = None
            for name, t, _ in self.transformers:
                if name == trans:
                    trans_obj = t
                    break
            if trans_obj is None:
                raise RuntimeError(f"Transformer '{trans}' not found")
            parts.append(trans_obj.transform(X[:, cols]))

        if self._remainder_idx and len(self._remainder_idx) > 0:
            parts.append(X[:, self._remainder_idx])

        if not parts:
            return np.zeros((X.shape[0], 0), dtype=float)
        return np.column_stack(parts) if len(parts) > 1 else parts[0]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
