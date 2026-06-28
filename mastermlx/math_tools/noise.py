from __future__ import annotations

import numpy as np


def _to_arr(x):
    return np.asarray(x, dtype=float)


# ---------------------------------------------------------------------------
# Additive noise
# ---------------------------------------------------------------------------


def gauss(x, scale=0.1, random_state=None):
    """Add zero-mean Gaussian noise."""
    x = _to_arr(x)
    rng = np.random.default_rng(random_state)
    return x + rng.normal(0.0, float(scale), size=x.shape)


def uniform(x, low=-0.1, high=0.1, random_state=None):
    """Add uniform noise in [low, high]."""
    x = _to_arr(x)
    rng = np.random.default_rng(random_state)
    return x + rng.uniform(float(low), float(high), size=x.shape)


def laplace(x, scale=0.1, random_state=None):
    """Add Laplace (double-exponential) noise, e.g. for differential privacy."""
    x = _to_arr(x)
    rng = np.random.default_rng(random_state)
    return x + rng.laplace(0.0, float(scale), size=x.shape)


# ---------------------------------------------------------------------------
# Multiplicative / replacement noise
# ---------------------------------------------------------------------------


def salt_pepper(x, prob=0.05, salt_val=None, pepper_val=None, random_state=None):
    """Randomly replace values with salt (high) or pepper (low)."""
    x = _to_arr(x)
    if salt_val is None:
        salt_val = np.max(x)
    if pepper_val is None:
        pepper_val = np.min(x)
    rng = np.random.default_rng(random_state)
    out = x.copy()
    mask = rng.random(size=x.shape)
    out[mask < float(prob) / 2] = salt_val
    out[(mask >= float(prob) / 2) & (mask < float(prob))] = pepper_val
    return out


def dropout(x, prob=0.2, random_state=None):
    """Randomly zero out elements (like Dropout, but for any array)."""
    x = _to_arr(x)
    rng = np.random.default_rng(random_state)
    mask = rng.random(size=x.shape) > float(prob)
    return np.where(mask, x, 0.0)


def poisson(x, random_state=None):
    """Apply Poisson (shot) noise, treating x as rate parameter."""
    x = _to_arr(x)
    x = np.maximum(x, 0.0)
    rng = np.random.default_rng(random_state)
    return rng.poisson(x).astype(float)


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------


def jitter(x, scale=0.01, random_state=None):
    """Add small Gaussian jitter, alias for small-scale gauss()."""
    return gauss(x, scale=scale, random_state=random_state)


def shuffle(x, axis=0, random_state=None):
    """Shuffle array along an axis."""
    x = _to_arr(x)
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(x.shape[axis])
    return np.take(x, idx, axis=axis)


def swap(x, prob=0.05, random_state=None):
    """Randomly swap adjacent pairs of elements along axis 0."""
    x = _to_arr(x)
    rng = np.random.default_rng(random_state)
    out = x.copy()
    n = x.shape[0]
    for i in range(n - 1):
        if rng.random() < float(prob):
            out[i], out[i + 1] = out[i + 1].copy(), out[i].copy()
    return out
