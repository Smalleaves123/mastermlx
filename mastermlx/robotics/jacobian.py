from __future__ import annotations

import numpy as np

from .kinematics import _geometric_jacobian_packed, _pack_links


def geometric_jacobian(links, joint_values=None, base=None, tool=None):
    """Return the geometric Jacobian of a serial manipulator."""

    links, a, alpha, d, theta, joint_type, offset = _pack_links(links)
    n = len(links)
    if joint_values is None:
        q = np.zeros(n, dtype=float)
    else:
        q = np.asarray(joint_values, dtype=float).reshape(-1)
        if q.size != n:
            raise ValueError(f"Expected {n} joint values, got {q.size}")
    return _geometric_jacobian_packed(a, alpha, d, theta, joint_type, offset, q, base=base, tool=tool)


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
