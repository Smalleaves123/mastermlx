from __future__ import annotations

import numpy as np

from .kinematics import _chain_transforms


def geometric_jacobian(links, joint_values=None, base=None, tool=None):
    """Return the geometric Jacobian of a serial manipulator."""

    T_end, frames, links = _chain_transforms(links, joint_values=joint_values, base=base, tool=tool)
    n = len(links)
    J = np.zeros((6, n), dtype=float)
    p_end = T_end[:3, 3]

    for i, link in enumerate(links):
        frame = frames[i]
        axis = frame[:3, 2]
        origin = frame[:3, 3]
        if link.joint_type.lower() == "revolute":
            J[:3, i] = np.cross(axis, p_end - origin)
            J[3:, i] = axis
        elif link.joint_type.lower() == "prismatic":
            J[:3, i] = axis
            J[3:, i] = 0.0
        else:
            raise ValueError("joint_type must be 'revolute' or 'prismatic'")
    return J


def planar_2r_jacobian(l1, l2, q1, q2):
    """Analytic Jacobian for a planar 2R arm."""

    s1 = np.sin(q1)
    c1 = np.cos(q1)
    s12 = np.sin(q1 + q2)
    c12 = np.cos(q1 + q2)
    return np.array(
        [
            [-l1 * s1 - l2 * s12, -l2 * s12],
            [l1 * c1 + l2 * c12, l2 * c12],
        ],
        dtype=float,
    )


def finite_difference_jacobian(func, x, eps=1e-6):
    """Numerically approximate the Jacobian of a vector-valued function."""

    x = np.asarray(x, dtype=float).reshape(-1)
    y0 = np.asarray(func(x), dtype=float).reshape(-1)
    J = np.zeros((y0.size, x.size), dtype=float)
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        y1 = np.asarray(func(x + dx), dtype=float).reshape(-1)
        y2 = np.asarray(func(x - dx), dtype=float).reshape(-1)
        J[:, i] = (y1 - y2) / (2.0 * eps)
    return J
