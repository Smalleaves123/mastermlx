from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils.validation import check_2d_array


def _yj_transform(x, lam):
    """Yeo-Johnson transform for a single lambda value."""
    x = np.asarray(x, dtype=float)
    pos = x >= 0
    out = np.empty_like(x, dtype=float)
    if np.isclose(lam, 0.0):
        out[pos] = np.log1p(x[pos])
    else:
        out[pos] = (np.power(x[pos] + 1.0, lam) - 1.0) / lam
    if np.isclose(lam, 2.0):
        out[~pos] = -np.log1p(-x[~pos])
    else:
        out[~pos] = -(np.power(-x[~pos] + 1.0, 2.0 - lam) - 1.0) / (2.0 - lam)
    return out


def _yj_inv(x, lam):
    """Inverse Yeo-Johnson transform."""
    x = np.asarray(x, dtype=float)
    pos = x >= 0
    out = np.empty_like(x, dtype=float)
    if np.isclose(lam, 0.0):
        out[pos] = np.expm1(x[pos])
    else:
        out[pos] = np.power(x[pos] * lam + 1.0, 1.0 / lam) - 1.0
    if np.isclose(lam, 2.0):
        out[~pos] = -np.expm1(-x[~pos])
    else:
        out[~pos] = -np.power(-x[~pos] * (2.0 - lam) + 1.0, 1.0 / (2.0 - lam)) + 1.0
    return out


class PowerTransform(BaseTransformer):
    """Yeo-Johnson power transform to make features more Gaussian."""

    def __init__(self, method="yeo-johnson"):
        if method not in {"yeo-johnson", "box-cox"}:
            raise ValueError("method must be 'yeo-johnson' or 'box-cox'")
        self.method = method
        self.lambdas_ = None

    def _fit_lambda(self, x):
        """Find lambda that maximizes log-likelihood for Yeo-Johnson."""
        x = np.asarray(x, dtype=float)
        n = x.size
        best_lam = 1.0
        best_ll = -np.inf
        for lam in np.linspace(-2.0, 2.0, 101):
            xt = _yj_transform(x, lam)
            if np.any(~np.isfinite(xt)):
                continue
            var = np.var(xt)
            if var <= 0:
                continue
            ll = -0.5 * n * np.log(var) + (lam - 1.0) * np.sum(np.sign(x) * np.log1p(np.abs(x)))
            if ll > best_ll:
                best_ll = ll
                best_lam = lam
        return best_lam

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        self.lambdas_ = np.array([self._fit_lambda(X[:, j]) for j in range(X.shape[1])])
        return self

    def transform(self, X):
        X = check_2d_array(X).astype(float)
        if self.lambdas_ is None:
            raise RuntimeError("Transform has not been fit yet")
        return np.column_stack([_yj_transform(X[:, j], self.lambdas_[j]) for j in range(X.shape[1])])

    def inverse_transform(self, X):
        X = check_2d_array(X).astype(float)
        if self.lambdas_ is None:
            raise RuntimeError("Transform has not been fit yet")
        return np.column_stack([_yj_inv(X[:, j], self.lambdas_[j]) for j in range(X.shape[1])])
