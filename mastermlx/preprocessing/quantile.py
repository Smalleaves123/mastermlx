from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils.math import _norm_ppf
from ..utils.validation import check_2d_array


def _quantile_map(x, n_quantiles, ref_vals):
    """Map x values to quantile-based reference values by rank interpolation."""
    x = np.asarray(x, dtype=float)
    ranks = np.argsort(np.argsort(x))
    frac = np.clip(ranks / max(x.size - 1, 1), 0.0, 1.0)
    idx = frac * (n_quantiles - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, n_quantiles - 1)
    w = idx - lo
    return (1.0 - w) * ref_vals[lo] + w * ref_vals[hi]


class QuantileTransform(BaseTransformer):
    """Map features to a uniform or normal distribution via quantiles."""

    def __init__(self, n_quantiles=1000, output_distribution="uniform", random_state=None):
        self.n_quantiles = int(n_quantiles)
        self.output_distribution = output_distribution
        self.random_state = random_state
        self.ref_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        dist = self.output_distribution
        if dist not in {"uniform", "normal"}:
            raise ValueError("output_distribution must be 'uniform' or 'normal'")
        nq = min(self.n_quantiles, X.shape[0])
        ref = np.linspace(0.0, 1.0, nq)
        if dist == "normal":
            ref = _norm_ppf(np.clip(ref, 1e-12, 1.0 - 1e-12))
        self.ref_ = ref
        return self

    def transform(self, X):
        X = check_2d_array(X).astype(float)
        if self.ref_ is None:
            raise RuntimeError("Transform has not been fit yet")
        nq = len(self.ref_)
        return np.column_stack([_quantile_map(X[:, j], nq, self.ref_) for j in range(X.shape[1])])
