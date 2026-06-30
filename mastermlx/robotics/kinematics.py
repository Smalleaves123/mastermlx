from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from ._robotics_ops import (
        forward_kinematics_dh as _cy_forward_kinematics_dh,
        geometric_jacobian_dh as _cy_geometric_jacobian_dh,
    )
except ImportError:  # pragma: no cover - fallback when Cython extensions are unavailable
    _cy_forward_kinematics_dh = None
    _cy_geometric_jacobian_dh = None


@dataclass(frozen=True)
class DHLink:
    """Denavit-Hartenberg link parameters.

    Parameters are interpreted using the standard DH convention.
    """

    a: float
    alpha: float
    d: float
    theta: float
    joint_type: str = "revolute"
    offset: float = 0.0


def dh_transform(a, alpha, d, theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _coerce_link(link):
    if isinstance(link, DHLink):
        return link
    if isinstance(link, dict):
        return DHLink(**link)
    if isinstance(link, (tuple, list)):
        if len(link) == 4:
            return DHLink(*link)
        if len(link) == 5:
            a, alpha, d, theta, joint_type = link
            return DHLink(a, alpha, d, theta, joint_type=joint_type)
    raise TypeError("Each link must be a DHLink, dict, or tuple/list of length 4 or 5")


def _pack_links(links):
    links = [_coerce_link(link) for link in links]
    n = len(links)
    a = np.empty(n, dtype=float)
    alpha = np.empty(n, dtype=float)
    d = np.empty(n, dtype=float)
    theta = np.empty(n, dtype=float)
    offset = np.empty(n, dtype=float)
    joint_type = np.empty(n, dtype=np.int8)
    for i, link in enumerate(links):
        a[i] = link.a
        alpha[i] = link.alpha
        d[i] = link.d
        theta[i] = link.theta
        offset[i] = link.offset
        joint_type[i] = 1 if link.joint_type.lower() == "revolute" else 0
    return links, a, alpha, d, theta, joint_type, offset


def _link_transform(link, q):
    joint_type = link.joint_type.lower()
    if joint_type == "revolute":
        return dh_transform(link.a, link.alpha, link.d, link.theta + q + link.offset)
    if joint_type == "prismatic":
        return dh_transform(link.a, link.alpha, link.d + q + link.offset, link.theta)
    raise ValueError("joint_type must be 'revolute' or 'prismatic'")


def _chain_transforms(links, joint_values=None, base=None, tool=None):
    links, a, alpha, d, theta, joint_type, offset = _pack_links(links)
    n = len(links)
    if joint_values is None:
        q = np.zeros(n, dtype=float)
    else:
        q = np.asarray(joint_values, dtype=float).reshape(-1)
        if q.size != n:
            raise ValueError(f"Expected {n} joint values, got {q.size}")

    if _cy_forward_kinematics_dh is not None:
        T, frames = _cy_forward_kinematics_dh(a, alpha, d, theta, joint_type, offset, q, base=base, tool=tool, return_frames=True)
        frames = [np.asarray(frame, dtype=float).copy() for frame in frames]
        return np.asarray(T, dtype=float), frames, links

    T = np.eye(4, dtype=float) if base is None else np.asarray(base, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 base transform, got {T.shape}")

    frames = [T.copy()]
    for link, qi in zip(links, q):
        T = T @ _link_transform(link, qi)
        frames.append(T.copy())
    if tool is not None:
        tool = np.asarray(tool, dtype=float)
        if tool.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 tool transform, got {tool.shape}")
        T = T @ tool
        frames.append(T.copy())
    return T, frames, links


def forward_kinematics(links, joint_values=None, base=None, tool=None, return_all=False):
    """Compute the forward kinematics of a serial chain."""

    T, frames, _ = _chain_transforms(links, joint_values=joint_values, base=base, tool=tool)
    if return_all:
        return T, frames
    return T


def chain_positions(links, joint_values=None, base=None, tool=None):
    """Return the position of each chain frame origin, including the base."""

    _, frames, _ = _chain_transforms(links, joint_values=joint_values, base=base, tool=tool)
    return np.stack([frame[:3, 3] for frame in frames], axis=0)


def _forward_kinematics_packed(a, alpha, d, theta, joint_type, offset, q, base=None, tool=None, return_all=False):
    if _cy_forward_kinematics_dh is not None:
        result = _cy_forward_kinematics_dh(a, alpha, d, theta, joint_type, offset, q, base=base, tool=tool, return_frames=return_all)
        if return_all:
            T, frames = result
            frames = [np.asarray(frame, dtype=float).copy() for frame in frames]
            return np.asarray(T, dtype=float), frames
        return np.asarray(result, dtype=float)

    n = a.size
    links = [
        DHLink(a=float(a[i]), alpha=float(alpha[i]), d=float(d[i]), theta=float(theta[i]), joint_type="revolute" if int(joint_type[i]) != 0 else "prismatic", offset=float(offset[i]))
        for i in range(n)
    ]
    T, frames, _ = _chain_transforms(links, joint_values=q, base=base, tool=tool)
    if return_all:
        return T, frames
    return T


def _geometric_jacobian_packed(a, alpha, d, theta, joint_type, offset, q, base=None, tool=None):
    if _cy_geometric_jacobian_dh is not None:
        return _cy_geometric_jacobian_dh(a, alpha, d, theta, joint_type, offset, q, base=base, tool=tool)

    links = [
        DHLink(a=float(a[i]), alpha=float(alpha[i]), d=float(d[i]), theta=float(theta[i]), joint_type="revolute" if int(joint_type[i]) != 0 else "prismatic", offset=float(offset[i]))
        for i in range(a.size)
    ]
    T_end, frames, links = _chain_transforms(links, joint_values=q, base=base, tool=tool)
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


def inverse_kinematics(
    target,
    links,
    joint_values=None,
    base=None,
    tool=None,
    max_iter=100,
    tol=1e-6,
    damping=1e-4,
    step_size=1.0,
):
    """Solve inverse kinematics with damped least squares.

    If `target` is a 3-vector, only position is matched. If it is a 4x4
    homogeneous transform, both position and an approximate orientation
    error are matched.
    """

    links, a, alpha, d, theta, joint_type, offset = _pack_links(links)
    n = len(links)
    if joint_values is None:
        q = np.zeros(n, dtype=float)
    else:
        q = np.asarray(joint_values, dtype=float).reshape(-1)
        if q.size != n:
            raise ValueError(f"Expected {n} joint values, got {q.size}")

    target = np.asarray(target, dtype=float)
    if target.shape == (3,):
        target_pos = target
        pose_target = None
    elif target.shape == (4, 4):
        target_pos = target[:3, 3]
        pose_target = target
    else:
        raise ValueError("target must be a 3-vector or 4x4 transform")

    for _ in range(int(max_iter)):
        T = _forward_kinematics_packed(a, alpha, d, theta, joint_type, offset, q, base=base, tool=tool, return_all=False)
        pos_err = target_pos - T[:3, 3]

        if pose_target is None:
            error = pos_err
            J = _geometric_jacobian_packed(a, alpha, d, theta, joint_type, offset, q, base=base, tool=tool)[:3]
        else:
            R_err = pose_target[:3, :3] @ T[:3, :3].T
            rot_err = 0.5 * np.array(
                [
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1],
                ],
                dtype=float,
            )
            error = np.concatenate([pos_err, rot_err])
            J = _geometric_jacobian_packed(a, alpha, d, theta, joint_type, offset, q, base=base, tool=tool)

        if np.linalg.norm(error) <= tol:
            return q

        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (damping**2) * np.eye(JJt.shape[0], dtype=float), error)
        q = q + step_size * dq

    return q
