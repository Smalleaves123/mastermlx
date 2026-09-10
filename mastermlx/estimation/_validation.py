from __future__ import annotations

import numpy as np


def as_vector(value, name, *, size=None):
    out = np.asarray(value, dtype=float).reshape(-1)
    if out.size < 1:
        raise ValueError(f"{name} must be non-empty")
    if size is not None and out.size != size:
        raise ValueError(f"{name} must have size {size}")
    if np.any(~np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def as_matrix(value, name, *, shape=None):
    out = np.asarray(value, dtype=float)
    if out.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if shape is not None and out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if np.any(~np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def as_square_matrix(value, name, *, size=None):
    out = as_matrix(value, name)
    expected = out.shape[0] if size is None else size
    if expected < 1 or out.shape != (expected, expected):
        raise ValueError(f"{name} must have shape ({expected}, {expected})")
    return out


def joseph_covariance(P, gain, H, R):
    identity_minus_gain = np.eye(P.shape[0], dtype=float) - gain @ H
    covariance = (
        identity_minus_gain @ P @ identity_minus_gain.T
        + gain @ R @ gain.T
    )
    return 0.5 * (covariance + covariance.T)
