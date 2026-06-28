from __future__ import annotations

import numpy as np


def sigmoid(x):
    x = np.asarray(x)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def _norm_ppf(p):
    """Normal quantile (Abramowitz & Stegun 26.2.23)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    sign = np.where(p < 0.5, -1.0, 1.0)
    q = np.where(p < 0.5, p, 1.0 - p)
    t = np.sqrt(-2.0 * np.log(q))
    c = np.array([2.515517, 0.802853, 0.010328])
    d = np.array([1.432788, 0.189269, 0.001308])
    num = c[0] + c[1] * t + c[2] * t ** 2
    den = 1.0 + d[0] * t + d[1] * t ** 2 + d[2] * t ** 3
    return sign * (t - num / den)


def log_sum_exp(a, axis=None):
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if axis is None:
        return float(out)
    return np.squeeze(out, axis=axis)
