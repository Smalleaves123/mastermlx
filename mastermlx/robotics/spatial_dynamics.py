"""NumPy reference dynamics for general serial URDF chains."""

from __future__ import annotations

import numpy as np

from .dynamics import normalize_link_inertias


def _inputs(robot, link_inertias):
    n_bodies = len(robot.chain.joints)
    values = robot.chain.link_inertias if link_inertias is None else link_inertias
    if values is None or len(values) != n_bodies or any(value is None for value in values):
        raise ValueError(
            "spatial dynamics requires one complete LinkInertia for every URDF child link"
        )
    return normalize_link_inertias(values, n_bodies)


def _validate_q(robot, joint_values):
    return robot.validate_joint_values(
        joint_values, check_limits=robot.joint_limits is not None
    )


def _body_jacobians(robot, joint_values, body_index, inertias):
    q = _validate_q(robot, joint_values)
    _, frames, origins, axes = robot.chain._forward_with_geometry(
        q, base=robot.base, tool=None
    )
    body_transform = frames[body_index + 1]
    inertia = inertias[body_index]
    rotation = body_transform[:3, :3]
    position = body_transform[:3, 3] + rotation @ np.asarray(inertia.center_of_mass)
    linear = np.zeros((3, robot.n_joints), dtype=float)
    angular = np.zeros((3, robot.n_joints), dtype=float)
    active_indices = [
        index for index, joint in enumerate(robot.chain.joints)
        if joint.joint_type != "fixed" and index <= body_index
    ]
    for active_index, path_index in enumerate(active_indices):
        joint = robot.chain.joints[path_index]
        if joint.joint_type in {"revolute", "continuous"}:
            linear[:, active_index] = np.cross(
                axes[active_index], position - origins[active_index]
            )
            angular[:, active_index] = axes[active_index]
        else:
            linear[:, active_index] = axes[active_index]
    return linear, angular, rotation


def _mass_and_gravity(robot, joint_values, inertias, gravity):
    matrix = np.zeros((robot.n_joints, robot.n_joints), dtype=float)
    forces = np.zeros(robot.n_joints, dtype=float)
    for body_index, inertia in enumerate(inertias):
        linear, angular, rotation = _body_jacobians(
            robot, joint_values, body_index, inertias
        )
        world_inertia = rotation @ np.asarray(inertia.inertia) @ rotation.T
        matrix += inertia.mass * linear.T @ linear + angular.T @ world_inertia @ angular
        forces -= linear.T @ (inertia.mass * gravity)
    return 0.5 * (matrix + matrix.T), forces


def spatial_mass_matrix(robot, joint_values=None, *, link_inertias=None):
    """Return ``M(q)`` for a general serial URDF chain."""

    q = _validate_q(robot, joint_values)
    inertias = _inputs(robot, link_inertias)
    return _mass_and_gravity(robot, q, inertias, np.zeros(3))[0]


def spatial_gravity_forces(
    robot, joint_values=None, *, gravity=(0.0, 0.0, -9.81), link_inertias=None
):
    """Return generalized gravity forces for a spatial URDF chain."""

    q = _validate_q(robot, joint_values)
    gravity = np.asarray(gravity, dtype=float).reshape(-1)
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
        raise ValueError("gravity must be a finite 3-vector")
    return _mass_and_gravity(robot, q, _inputs(robot, link_inertias), gravity)[1]


def spatial_coriolis_forces(
    robot,
    joint_values,
    joint_velocities,
    *,
    link_inertias=None,
    epsilon=1e-6,
):
    """Return Coriolis and centrifugal forces from finite-difference ``M(q)``."""

    q = _validate_q(robot, joint_values)
    qd = np.asarray(joint_velocities, dtype=float).reshape(-1)
    if qd.shape != (robot.n_joints,) or not np.all(np.isfinite(qd)):
        raise ValueError(f"joint_velocities must have shape ({robot.n_joints},)")
    epsilon = float(epsilon)
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be a positive finite value")
    inertias = _inputs(robot, link_inertias)
    derivatives = np.empty((robot.n_joints, robot.n_joints, robot.n_joints), dtype=float)
    for coordinate in range(robot.n_joints):
        delta = np.zeros(robot.n_joints, dtype=float)
        delta[coordinate] = epsilon
        plus = _mass_and_gravity(robot, q + delta, inertias, np.zeros(3))[0]
        minus = _mass_and_gravity(robot, q - delta, inertias, np.zeros(3))[0]
        derivatives[:, :, coordinate] = (plus - minus) / (2.0 * epsilon)
    result = np.zeros(robot.n_joints, dtype=float)
    for row in range(robot.n_joints):
        for first in range(robot.n_joints):
            for second in range(robot.n_joints):
                result[row] += 0.5 * (
                    derivatives[row, first, second]
                    + derivatives[row, second, first]
                    - derivatives[first, second, row]
                ) * qd[first] * qd[second]
    return result


def spatial_inverse_dynamics(
    robot,
    joint_values,
    joint_velocities,
    joint_accelerations,
    *,
    gravity=(0.0, 0.0, -9.81),
    link_inertias=None,
    include_coriolis=True,
):
    """Return ``M(q)qdd + C(q,qd) + g(q)``."""

    q = _validate_q(robot, joint_values)
    qd = np.asarray(joint_velocities, dtype=float).reshape(-1)
    qdd = np.asarray(joint_accelerations, dtype=float).reshape(-1)
    if qd.shape != q.shape or qdd.shape != q.shape:
        raise ValueError("joint velocities and accelerations must match joint_values")
    if not np.all(np.isfinite(qd)) or not np.all(np.isfinite(qdd)):
        raise ValueError("joint velocities and accelerations must be finite")
    inertias = _inputs(robot, link_inertias)
    gravity = np.asarray(gravity, dtype=float).reshape(-1)
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
        raise ValueError("gravity must be a finite 3-vector")
    matrix, gravity_term = _mass_and_gravity(robot, q, inertias, gravity)
    result = matrix @ qdd + gravity_term
    if include_coriolis:
        result += spatial_coriolis_forces(
            robot, q, qd, link_inertias=inertias
        )
    return result


def spatial_forward_dynamics(
    robot,
    joint_values,
    joint_velocities,
    joint_torques,
    *,
    gravity=(0.0, 0.0, -9.81),
    link_inertias=None,
    include_coriolis=True,
):
    """Return joint accelerations from applied torques."""

    q = _validate_q(robot, joint_values)
    qd = np.asarray(joint_velocities, dtype=float).reshape(-1)
    torques = np.asarray(joint_torques, dtype=float).reshape(-1)
    if qd.shape != q.shape or torques.shape != q.shape:
        raise ValueError("joint velocities and torques must match joint_values")
    if not np.all(np.isfinite(qd)) or not np.all(np.isfinite(torques)):
        raise ValueError("joint velocities and torques must be finite")
    inertias = _inputs(robot, link_inertias)
    gravity = np.asarray(gravity, dtype=float).reshape(-1)
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
        raise ValueError("gravity must be a finite 3-vector")
    matrix, gravity_term = _mass_and_gravity(robot, q, inertias, gravity)
    rhs = torques - gravity_term
    if include_coriolis:
        rhs -= spatial_coriolis_forces(robot, q, qd, link_inertias=inertias)
    return np.linalg.solve(matrix, rhs)


def spatial_computed_torque(
    robot,
    joint_values,
    joint_velocities,
    desired_positions,
    desired_velocities=None,
    desired_accelerations=None,
    *,
    kp=25.0,
    kd=8.0,
    gravity=(0.0, 0.0, -9.81),
    link_inertias=None,
    torque_limits=None,
):
    """Return a gravity/Coriolis-compensated computed-torque command."""

    q = _validate_q(robot, joint_values)
    qd = np.asarray(joint_velocities, dtype=float).reshape(-1)
    target = np.asarray(desired_positions, dtype=float).reshape(-1)
    target_velocity = np.zeros(robot.n_joints) if desired_velocities is None else np.asarray(desired_velocities, dtype=float).reshape(-1)
    target_acceleration = np.zeros(robot.n_joints) if desired_accelerations is None else np.asarray(desired_accelerations, dtype=float).reshape(-1)
    if any(value.shape != (robot.n_joints,) for value in (qd, target, target_velocity, target_acceleration)):
        raise ValueError("computed-torque vectors must match robot.n_joints")
    kp = np.asarray(kp, dtype=float)
    kd = np.asarray(kd, dtype=float)
    if np.any(~np.isfinite(kp)) or np.any(~np.isfinite(kd)) or np.any(kp <= 0.0) or np.any(kd <= 0.0):
        raise ValueError("kp and kd must be positive and finite")
    qdd_command = target_acceleration + kp * (target - q) + kd * (target_velocity - qd)
    torque = spatial_inverse_dynamics(
        robot, q, qd, qdd_command, gravity=gravity,
        link_inertias=link_inertias, include_coriolis=True,
    )
    if torque_limits is not None:
        limits = np.asarray(torque_limits, dtype=float)
        if limits.ndim == 0:
            limits = np.full(robot.n_joints, float(limits))
        limits = limits.reshape(-1)
        if limits.shape != (robot.n_joints,) or not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
            raise ValueError("torque_limits must be positive and match robot.n_joints")
        torque = np.clip(torque, -limits, limits)
    return torque


__all__ = [
    "spatial_computed_torque",
    "spatial_coriolis_forces",
    "spatial_forward_dynamics",
    "spatial_gravity_forces",
    "spatial_inverse_dynamics",
    "spatial_mass_matrix",
]
