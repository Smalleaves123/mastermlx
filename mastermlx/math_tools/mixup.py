from __future__ import annotations

import numpy as np


def mixup(X, y, alpha=0.2, random_state=None):
    """Mixup augmentation: convex combinations of pairs.

    Returns (X_mixed, y_mixed) where each row is a blend of two
    randomly paired samples.  Labels can be 1D (classification
    integer labels are returned as two columns) or 1D float (regression).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.size:
        raise ValueError("X and y must have the same number of rows")
    n = X.shape[0]
    rng = np.random.default_rng(random_state)

    lam = rng.beta(float(alpha), float(alpha), size=n)
    lam = np.maximum(lam, 1.0 - lam)  # keep mixing weight ≥ 0.5
    idx = rng.permutation(n)

    X_mix = lam[:, None] * X + (1.0 - lam[:, None]) * X[idx]
    y_mix = np.column_stack([lam, 1.0 - lam])  # return sample weights

    return X_mix, y[idx], y_mix


def cutmix(X, y, alpha=1.0, random_state=None):
    """CutMix: replace a patch of one sample with another.

    Returns (X_mixed, y_a, y_b, lam) for use with weighted loss.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n = X.shape[0]
    rng = np.random.default_rng(random_state)
    d = X.shape[1]

    lam_vals = rng.beta(float(alpha), float(alpha), size=n)
    lam_vals = np.maximum(lam_vals, 1.0 - lam_vals)
    idx = rng.permutation(n)

    X_mix = X.copy()
    for i in range(n):
        n_feat = max(1, int(d * (1.0 - lam_vals[i])))
        start = rng.integers(0, d - n_feat + 1)
        X_mix[i, start:start + n_feat] = X[idx[i], start:start + n_feat]

    return X_mix, y[idx], lam_vals
