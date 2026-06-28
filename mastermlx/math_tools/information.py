from __future__ import annotations

import numpy as np


def empirical_distribution(x):
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("x must be non-empty")
    values, counts = np.unique(x, return_counts=True)
    probs = counts.astype(float) / counts.sum()
    return values, probs


def _sample_entropy(x, base=np.e):
    return entropy(empirical_distribution(x)[1], base=base)


def _normalize_probabilities(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        p = p.ravel()
    if p.size == 0:
        raise ValueError("probability vector must be non-empty")
    total = float(np.sum(p))
    if total <= 0:
        raise ValueError("probabilities must sum to a positive value")
    p = p / total
    if np.any(p < 0):
        raise ValueError("probabilities must be non-negative")
    return p


def _log_base(base):
    if base is None or base == np.e:
        return 1.0
    base = float(base)
    if base <= 0 or base == 1.0:
        raise ValueError("base must be positive and different from 1")
    return np.log(base)


def entropy(p, base=np.e):
    p = _normalize_probabilities(p)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(p[mask])) / _log_base(base))


def entropy_from_counts(counts, base=np.e):
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1:
        raise ValueError("counts must be a 1D array")
    return entropy(counts, base=base)


def cross_entropy(p, q, base=np.e):
    p = _normalize_probabilities(p)
    q = _normalize_probabilities(q)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    q = np.clip(q, 1e-12, 1.0)
    return float(-np.sum(p * np.log(q)) / _log_base(base))


def kl_divergence(p, q, base=np.e):
    p = _normalize_probabilities(p)
    q = _normalize_probabilities(q)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    mask = p > 0
    q = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])) / _log_base(base))


def js_divergence(p, q, base=np.e):
    p = _normalize_probabilities(p)
    q = _normalize_probabilities(q)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, base=base) + 0.5 * kl_divergence(q, m, base=base)


def joint_entropy(x, y, base=np.e):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length")
    pairs = np.stack([x, y], axis=1)
    _, counts = np.unique(pairs, axis=0, return_counts=True)
    return entropy(counts, base=base)


def conditional_entropy(x, y, base=np.e):
    return float(joint_entropy(x, y, base=base) - _sample_entropy(y, base=base))


def mutual_information(x, y, base=np.e):
    return float(_sample_entropy(x, base=base) + _sample_entropy(y, base=base) - joint_entropy(x, y, base=base))


def variation_of_information(x, y, base=np.e):
    hx = _sample_entropy(x, base=base)
    hy = _sample_entropy(y, base=base)
    mi = mutual_information(x, y, base=base)
    return float(hx + hy - 2.0 * mi)


def normalized_mutual_information(x, y, base=np.e, method="sqrt"):
    hx = _sample_entropy(x, base=base)
    hy = _sample_entropy(y, base=base)
    mi = mutual_information(x, y, base=base)
    method = method.lower()
    if method == "sqrt":
        denom = np.sqrt(hx * hy)
    elif method == "arithmetic":
        denom = 0.5 * (hx + hy)
    elif method == "max":
        denom = max(hx, hy)
    elif method == "min":
        denom = min(hx, hy)
    else:
        raise ValueError("method must be one of: sqrt, arithmetic, max, min")
    return float(mi / denom) if denom > 0 else 0.0
