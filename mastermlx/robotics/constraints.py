"""Joint-space constraint helpers for serial robot models."""

from __future__ import annotations

import numpy as np


def validate_joint_limits(joint_limits, n_joints):
    """Normalize position limits to an ``(n_joints, 2)`` float array."""

    if joint_limits is None:
        return None
    limits = np.asarray(joint_limits, dtype=float)
    if limits.shape != (int(n_joints), 2):
        raise ValueError(f"joint_limits must have shape ({int(n_joints)}, 2)")
    if not np.all(np.isfinite(limits)) or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("joint_limits must contain finite lower < upper bounds")
    return limits.copy()


def check_joint_limits(joint_values, joint_limits, *, name="joint_values"):
    """Raise ``ValueError`` when joint values exceed configured limits."""

    if joint_limits is None:
        return
    values = np.asarray(joint_values, dtype=float)
    if values.ndim < 1 or values.shape[-1] != joint_limits.shape[0]:
        raise ValueError(f"{name} must end with {joint_limits.shape[0]} joint values")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(values < joint_limits[:, 0]) or np.any(values > joint_limits[:, 1]):
        raise ValueError(f"{name} exceeds configured joint_limits")


def clip_joint_values(joint_values, joint_limits):
    """Project one or more joint configurations into position limits."""

    values = np.asarray(joint_values, dtype=float)
    if joint_limits is None:
        return values.copy()
    if values.ndim < 1 or values.shape[-1] != joint_limits.shape[0]:
        raise ValueError(f"joint_values must end with {joint_limits.shape[0]} joint values")
    if not np.all(np.isfinite(values)):
        raise ValueError("joint_values must contain only finite values")
    return np.clip(values, joint_limits[:, 0], joint_limits[:, 1])


def joint_limit_violation(joint_values, joint_limits):
    """Return the maximum absolute position-limit violation per configuration."""

    values = np.asarray(joint_values, dtype=float)
    if joint_limits is None:
        return np.zeros(values.shape[:-1], dtype=float)
    if values.ndim < 1 or values.shape[-1] != joint_limits.shape[0]:
        raise ValueError(f"joint_values must end with {joint_limits.shape[0]} joint values")
    lower = np.maximum(joint_limits[:, 0] - values, 0.0)
    upper = np.maximum(values - joint_limits[:, 1], 0.0)
    return np.max(np.maximum(lower, upper), axis=-1)


__all__ = [
    "check_joint_limits",
    "clip_joint_values",
    "joint_limit_violation",
    "validate_joint_limits",
]
