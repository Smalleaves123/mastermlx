from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils.math import _norm_ppf
from ..utils.validation import check_2d_array


class QuantileTransform(BaseTransformer):
    """Map features to a uniform or normal distribution via quantiles."""

    def __init__(self, n_quantiles=1000, output_distribution="uniform", random_state=None):
        self.n_quantiles = int(n_quantiles)
        self.output_distribution = output_distribution
        self.random_state = random_state
        self.ref_ = None
        self.quantiles_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        dist = self.output_distribution
        if dist not in {"uniform", "normal"}:
            raise ValueError("output_distribution must be 'uniform' or 'normal'")
        if self.n_quantiles < 1:
            raise ValueError("n_quantiles must be at least 1")
        nq = min(self.n_quantiles, X.shape[0])
        self.ref_ = np.linspace(0.0, 1.0, nq)
        self.quantiles_ = np.quantile(X, self.ref_, axis=0)
        self._set_n_features(X)
        return self

    def transform(self, X):
        X = self._check_X(X, dtype=float)
        if self.ref_ is None or self.quantiles_ is None:
            raise RuntimeError("Transform has not been fit yet")
        uniform = np.column_stack([
            np.interp(X[:, j], self.quantiles_[:, j], self.ref_)
            for j in range(X.shape[1])
        ])
        if self.output_distribution == "uniform":
            return uniform
        return _norm_ppf(np.clip(uniform, 1e-12, 1.0 - 1e-12))
