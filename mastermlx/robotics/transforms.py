from __future__ import annotations

import numpy as np


def _as_vector3(v):
    arr = np.asarray(v, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"Expected a 3-vector, got shape {np.shape(v)}")
    return arr


def skew(v):
    x, y, z = _as_vector3(v)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def unskew(M):
    M = np.asarray(M, dtype=float)
    if M.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 matrix, got {M.shape}")
    return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=float)


def rot_x(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def rot_y(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def rot_z(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rpy_to_matrix(roll, pitch, yaw):
    """Convert roll-pitch-yaw angles to a rotation matrix.

    The convention is ZYX composition: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """

    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def euler_to_matrix(roll, pitch, yaw):
    return rpy_to_matrix(roll, pitch, yaw)


def matrix_to_euler(R):
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 matrix, got {R.shape}")

    sy = -np.clip(R[2, 0], -1.0, 1.0)
    pitch = np.arcsin(sy)
    cy = np.cos(pitch)
    if abs(cy) > 1e-12:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=float)


def quaternion_to_matrix(q):
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != 4:
        raise ValueError(f"Expected a 4-vector (w, x, y, z), got shape {np.shape(q)}")
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Quaternion must be non-zero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quaternion(R):
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 matrix, got {R.shape}")
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def homogeneous_transform(rotation, translation):
    R = np.asarray(rotation, dtype=float)
    t = _as_vector3(translation)
    if R.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got {R.shape}")
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_transform(T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {T.shape}")
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4, dtype=float)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def compose_transform(*transforms):
    T = np.eye(4, dtype=float)
    for transform in transforms:
        Ti = np.asarray(transform, dtype=float)
        if Ti.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 transform, got {Ti.shape}")
        T = T @ Ti
    return T


def transform_points(T, points):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {T.shape}")
    pts = np.asarray(points, dtype=float)
    if pts.shape[-1] != 3:
        raise ValueError("Points must have shape (..., 3)")
    pts_flat = pts.reshape(-1, 3)
    hom = np.concatenate([pts_flat, np.ones((pts_flat.shape[0], 1), dtype=float)], axis=1)
    out = (T @ hom.T).T[:, :3]
    return out.reshape(pts.shape)
