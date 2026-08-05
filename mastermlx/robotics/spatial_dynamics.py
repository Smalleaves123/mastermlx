"""NumPy reference dynamics for general serial URDF chains."""

from __future__ import annotations

import numpy as np

from ..config import get_backend
from .dynamics import normalize_link_inertias
from .urdf_parser import _load_cpp_spatial


def _inputs(robot, link_inertias):
    n_bodies = len(robot.chain.joints)
    values = robot.chain.link_inertias if link_inertias is None else link_inertias
    if values is None or len(values) != n_bodies or any(value is None for value in values):
        raise ValueError(
            "spatial dynamics requires one complete LinkInertia for every URDF child link"
        )
    return normalize_link_inertias(values, n_bodies)


def _compiled_batch(
    robot, inertias, joint_values, joint_velocities, gravity, epsilon, compute_coriolis
):
    cpp = _load_cpp_spatial(get_backend())
    if cpp is None or not callable(getattr(cpp, "spatial_dynamics_batch_urdf", None)):
        return None
    origin_xyz, origin_rpy, axes, joint_types = robot.chain._compiled_arrays()
    masses = np.ascontiguousarray([inertia.mass for inertia in inertias], dtype=float)
    center_of_mass = np.ascontiguousarray(
        [inertia.center_of_mass for inertia in inertias], dtype=float
    )
    inertia_matrices = np.ascontiguousarray([inertia.inertia for inertia in inertias], dtype=float)
    values = np.ascontiguousarray(np.asarray(joint_values, dtype=float))
    velocities = np.ascontiguousarray(np.asarray(joint_velocities, dtype=float))
    result = cpp.spatial_dynamics_batch_urdf(
        origin_xyz,
        origin_rpy,
        axes,
        joint_types,
        masses,
        center_of_mass,
        inertia_matrices,
        values,
        velocities,
        np.ascontiguousarray(gravity, dtype=float),
        float(epsilon),
        bool(compute_coriolis),
        base=getattr(robot, "base", None),
    )
    return tuple(np.asarray(value, dtype=float) for value in result)


def _compiled_inverse_batch(
    robot, inertias, joint_values, joint_velocities, joint_accelerations, gravity
):
    cpp = _load_cpp_spatial(get_backend())
    if cpp is None or not callable(getattr(cpp, "inverse_dynamics_batch_urdf", None)):
        return None
    origin_xyz, origin_rpy, axes, joint_types = robot.chain._compiled_arrays()
    masses = np.ascontiguousarray([inertia.mass for inertia in inertias], dtype=float)
    center_of_mass = np.ascontiguousarray(
        [inertia.center_of_mass for inertia in inertias], dtype=float
    )
    inertia_matrices = np.ascontiguousarray([inertia.inertia for inertia in inertias], dtype=float)
    return np.asarray(
        cpp.inverse_dynamics_batch_urdf(
            origin_xyz,
            origin_rpy,
            axes,
            joint_types,
            masses,
            center_of_mass,
            inertia_matrices,
            np.ascontiguousarray(joint_values, dtype=float),
            np.ascontiguousarray(joint_velocities, dtype=float),
            np.ascontiguousarray(joint_accelerations, dtype=float),
            np.ascontiguousarray(gravity, dtype=float),
            base=getattr(robot, "base", None),
        ),
        dtype=float,
    )


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


def _validate_batch_values(robot, values, name, *, check_limits=True):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != robot.n_joints or values.shape[0] < 1:
        raise ValueError(f"{name} must have shape (n_samples, {robot.n_joints})")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    if check_limits and robot.joint_limits is not None:
        for row in values:
            robot.validate_joint_values(row, check_limits=True)
    return np.ascontiguousarray(values)


def spatial_dynamics_batch(
    robot,
    joint_values,
    joint_velocities,
    *,
    gravity=(0.0, 0.0, -9.81),
    link_inertias=None,
    epsilon=1e-6,
    compute_coriolis=True,
):
    """Return batched ``M(q)``, gravity, and Coriolis terms for a URDF chain."""

    values = _validate_batch_values(robot, joint_values, "joint_values")
    velocities = _validate_batch_values(
        robot, joint_velocities, "joint_velocities", check_limits=False
    )
    if velocities.shape != values.shape:
        raise ValueError("joint_velocities must have the same shape as joint_values")
    gravity = np.asarray(gravity, dtype=float).reshape(-1)
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
        raise ValueError("gravity must be a finite 3-vector")
    epsilon = float(epsilon)
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be a positive finite value")
    inertias = _inputs(robot, link_inertias)
    compiled = _compiled_batch(
        robot, inertias, values, velocities, gravity, epsilon, compute_coriolis
    )
    if compiled is not None:
        return compiled
    samples = values.shape[0]
    matrices = np.empty((samples, robot.n_joints, robot.n_joints), dtype=float)
    forces = np.empty((samples, robot.n_joints), dtype=float)
    coriolis = np.zeros_like(forces)
    for index, (configuration, velocity) in enumerate(zip(values, velocities)):
        matrices[index], forces[index] = _mass_and_gravity(
            robot, configuration, inertias, gravity
        )
        if compute_coriolis:
            coriolis[index] = spatial_coriolis_forces(
                robot,
                configuration,
                velocity,
                link_inertias=inertias,
                epsilon=epsilon,
            )
    return matrices, forces, coriolis


def spatial_inverse_dynamics_batch(
    robot,
    joint_values,
    joint_velocities,
    joint_accelerations,
    *,
    gravity=(0.0, 0.0, -9.81),
    link_inertias=None,
    include_coriolis=True,
):
    """Return inverse-dynamics torques for a batch of general URDF states."""

    values = _validate_batch_values(robot, joint_values, "joint_values")
    velocities = _validate_batch_values(
        robot, joint_velocities, "joint_velocities", check_limits=False
    )
    accelerations = np.asarray(joint_accelerations, dtype=float)
    if velocities.shape != values.shape or accelerations.shape != values.shape:
        raise ValueError("joint velocities and accelerations must match joint_values")
    if not np.all(np.isfinite(accelerations)):
        raise ValueError("joint_accelerations must contain only finite values")
    gravity = np.asarray(gravity, dtype=float).reshape(-1)
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
        raise ValueError("gravity must be a finite 3-vector")
    inertias = _inputs(robot, link_inertias)
    compiled = _compiled_inverse_batch(
        robot,
        inertias,
        values,
        velocities if include_coriolis else np.zeros_like(velocities),
        accelerations,
        gravity,
    )
    if compiled is not None:
        return compiled
    matrices, forces, coriolis = spatial_dynamics_batch(
        robot,
        values,
        velocities,
        gravity=gravity,
        link_inertias=inertias,
        compute_coriolis=include_coriolis,
    )
    result = np.einsum("nij,nj->ni", matrices, accelerations) + forces
    if include_coriolis:
        result += coriolis
    return result


def spatial_mass_matrix(robot, joint_values=None, *, link_inertias=None):
    """Return ``M(q)`` for a general serial URDF chain."""

    q = _validate_q(robot, joint_values)
    inertias = _inputs(robot, link_inertias)
    compiled = _compiled_batch(
        robot, inertias, q[None, :], np.zeros((1, robot.n_joints)), np.zeros(3), 1e-6, False
    )
    if compiled is not None:
        return compiled[0][0]
    return _mass_and_gravity(robot, q, inertias, np.zeros(3))[0]


def spatial_gravity_forces(
    robot, joint_values=None, *, gravity=(0.0, 0.0, -9.81), link_inertias=None
):
    """Return generalized gravity forces for a spatial URDF chain."""

    q = _validate_q(robot, joint_values)
    gravity = np.asarray(gravity, dtype=float).reshape(-1)
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
        raise ValueError("gravity must be a finite 3-vector")
    inertias = _inputs(robot, link_inertias)
    compiled = _compiled_batch(
        robot, inertias, q[None, :], np.zeros((1, robot.n_joints)), gravity, 1e-6, False
    )
    if compiled is not None:
        return compiled[1][0]
    return _mass_and_gravity(robot, q, inertias, gravity)[1]


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
    compiled = _compiled_batch(
        robot, inertias, q[None, :], qd[None, :], np.zeros(3), epsilon, True
    )
    if compiled is not None:
        return compiled[2][0]
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
    compiled_inverse = _compiled_inverse_batch(
        robot,
        inertias,
        q[None, :],
        qd[None, :] if include_coriolis else np.zeros((1, robot.n_joints)),
        qdd[None, :],
        gravity,
    )
    if compiled_inverse is not None:
        return compiled_inverse[0]
    compiled = _compiled_batch(
        robot, inertias, q[None, :], qd[None, :], gravity, 1e-6, include_coriolis
    )
    if compiled is not None:
        matrix, gravity_term, coriolis = (value[0] for value in compiled)
        result = matrix @ qdd + gravity_term
        if include_coriolis:
            result += coriolis
        return result
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
    compiled = _compiled_batch(
        robot, inertias, q[None, :], qd[None, :], gravity, 1e-6, include_coriolis
    )
    if compiled is not None:
        matrix, gravity_term, coriolis = (value[0] for value in compiled)
        rhs = torques - gravity_term
        if include_coriolis:
            rhs -= coriolis
        return np.linalg.solve(matrix, rhs)
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
    "spatial_dynamics_batch",
    "spatial_forward_dynamics",
    "spatial_gravity_forces",
    "spatial_inverse_dynamics",
    "spatial_inverse_dynamics_batch",
    "spatial_mass_matrix",
]
