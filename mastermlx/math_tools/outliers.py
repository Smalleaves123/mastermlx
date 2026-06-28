from __future__ import annotations

import numpy as np

from ..utils.math import _norm_ppf


def _t_ppf(p, df):
    """Approximate t-distribution quantile via normal approximation."""
    z = _norm_ppf(p)
    nu = float(df)
    # Use a simple correction for small df
    return float(z * np.sqrt(nu / (nu - 2.0))) if nu > 2.0 else float(z)


def _to_1d(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        x = x.ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty")
    return x


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------


def zscore(x):
    x = _to_1d(x)
    mu = np.mean(x)
    s = np.std(x)
    s = max(s, 1e-12)
    return np.abs(x - mu) / s


def mod_zscore(x):
    x = _to_1d(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    mad = max(mad, 1e-12)
    return 0.6745 * np.abs(x - med) / mad


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


def zscore_outliers(x, thresh=3.0):
    scores = zscore(x)
    return np.flatnonzero(scores > float(thresh))


def mod_zscore_outliers(x, thresh=3.5):
    scores = mod_zscore(x)
    return np.flatnonzero(scores > float(thresh))


def iqr_outliers(x, factor=1.5):
    x = _to_1d(x)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lo = q1 - factor * iqr
    hi = q3 + factor * iqr
    return np.flatnonzero((x < lo) | (x > hi))


def grubbs(x, alpha=0.05):
    x = _to_1d(x)
    n = x.size
    if n < 3:
        raise ValueError("grubbs requires at least 3 samples")
    mu = np.mean(x)
    s = np.std(x, ddof=1)
    s = max(s, 1e-12)
    g = np.max(np.abs(x - mu)) / s
    t_val = _t_ppf(1.0 - alpha / (2.0 * n), n - 2)
    critical = (n - 1) * t_val / np.sqrt(n * (n - 2) + n * t_val ** 2)
    idx = int(np.argmax(np.abs(x - mu)))
    return idx, float(g), float(critical), g > critical
