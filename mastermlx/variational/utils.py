from __future__ import annotations

import math

import numpy as np

from ..utils.math import log_sum_exp


def digamma(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    work = x.copy()

    mask = work < 6.0
    while np.any(mask):
        out[mask] -= 1.0 / work[mask]
        work[mask] += 1.0
        mask = work < 6.0

    inv = 1.0 / work
    inv2 = inv * inv
    out += (
        np.log(work)
        - 0.5 * inv
        - inv2 / 12.0
        + inv2 * inv2 / 120.0
        - inv2 * inv2 * inv2 / 252.0
    )
    return out


def log_gamma(x):
    x = np.asarray(x, dtype=float)
    flat = [math.lgamma(float(v)) for v in x.ravel()]
    return np.asarray(flat, dtype=float).reshape(x.shape)


def normalize_log_probs(log_probs, axis=1):
    log_probs = np.asarray(log_probs, dtype=float)
    log_norm = log_sum_exp(log_probs, axis=axis)
    if axis == 1:
        probs = np.exp(log_probs - log_norm[:, None])
    elif axis == 0:
        probs = np.exp(log_probs - log_norm[None, :])
    else:
        raise ValueError("axis must be 0 or 1")
    return probs, log_norm


def has_converged(current, previous, tol):
    return previous is not None and abs(float(current) - float(previous)) < float(tol)
