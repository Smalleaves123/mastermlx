"""Rigid-body dynamics helpers for serial DH robot models.

The functions use per-link center-of-mass Jacobians, making the mass and
gravity terms easy to inspect and suitable as a reliable baseline before a
future recursive Newton-Euler compiled kernel is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib

import numpy as np

from ..config import get_backend
from .kinematics import _coerce_link, _pack_links_cached, forward_kinematics


@lru_cache(maxsize=3)
def _load_cpp_dynamics(backend=None):
    """Load the optional C++ rigid-body dynamics kernels for ``auto``."""

    if backend is None:
        backend = get_backend()
    if backend != "auto":
        return None
    try:
        return importlib.import_module("mastermlx.robotics._dynamics_cpp")
    except ImportError:
        return None


@dataclass(frozen=True)
class LinkInertia:
    """Mass properties of one DH link in its post-joint coordinate frame."""

    mass: float
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inertia: tuple[tuple[float, float, float], ...] | tuple[float, float, float] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )

    def __post_init__(self):
        mass = float(self.mass)
        center_of_mass = np.asarray(self.center_of_mass, dtype=float).reshape(-1)
        inertia = np.asarray(self.inertia, dtype=float)
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("mass must be a positive finite value")
        if center_of_mass.shape != (3,) or not np.all(np.isfinite(center_of_mass)):
            raise ValueError("center_of_mass must be a finite 3-vector")
        if inertia.shape == (3,):
            inertia = np.diag(inertia)
        if inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
            raise ValueError("inertia must be a finite 3-vector or 3x3 matrix")
        inertia = 0.5 * (inertia + inertia.T)
        if np.min(np.linalg.eigvalsh(inertia)) < -1e-12:
            raise ValueError("inertia must be positive semidefinite")
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "center_of_mass", tuple(float(value) for value in center_of_mass))
        object.__setattr__(
            self,
            "inertia",
            tuple(tuple(float(value) for value in row) for row in inertia),
        )


def normalize_link_inertias(link_inertias, n_links):
    """Validate and normalize mass properties for a serial chain."""

    if link_inertias is None:
        return None
    values = list(link_inertias)
    if len(values) != n_links:
        raise ValueError("link_inertias must contain one entry per link")
    normalized = []
    for value in values:
        if isinstance(value, LinkInertia):
            normalized.append(value)
        elif isinstance(value, dict):
            params = dict(value)
            if "com" in params and "center_of_mass" not in params:
                params["center_of_mass"] = params.pop("com")
            normalized.append(LinkInertia(**params))
        else:
            raise TypeError("link_inertias entries must be LinkInertia objects or dictionaries")
    return tuple(normalized)


def _normalized_inputs(links, link_inertias):
    links = [_coerce_link(link) for link in links]
    inertias = normalize_link_inertias(link_inertias, len(links))
    if inertias is None:
        raise ValueError("link_inertias are required for dynamics calculations")
    return links, inertias


def _joint_batch(values, n_joints, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != n_joints or values.shape[0] < 1:
        raise ValueError(f"{name} must have shape (n_samples, {n_joints})")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _output_buffer(output, shape, name):
    if output is None:
        return np.empty(shape, dtype=float)
    if not isinstance(output, np.ndarray) or output.dtype != np.dtype(float) or not output.flags.c_contiguous:
        raise ValueError(f"{name} must be a contiguous float64 NumPy array")
    if output.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return output


def _packed_dynamics_inputs(links, inertias):
    """Pack immutable DH and inertial data for the optional C++ kernel."""

    _, a, alpha, d, theta, joint_type, offset = _pack_links_cached(links)
    masses = np.asarray([inertia.mass for inertia in inertias], dtype=float)
    center_of_mass = np.asarray([inertia.center_of_mass for inertia in inertias], dtype=float)
    inertia_matrices = np.asarray([inertia.inertia for inertia in inertias], dtype=float)
    return a, alpha, d, theta, joint_type, offset, masses, center_of_mass, inertia_matrices


def _cpp_dynamics_arguments(links, inertias, values):
    return (*_packed_dynamics_inputs(links, inertias), np.ascontiguousarray(values, dtype=float))


def _body_jacobians(links, frames, body_index, inertia):
    n_joints = len(links)
    transform = np.asarray(frames[body_index + 1], dtype=float)
    rotation = transform[:3, :3]
    position = transform[:3, 3] + rotation @ np.asarray(inertia.center_of_mass, dtype=float)
    linear = np.zeros((3, n_joints), dtype=float)
    angular = np.zeros((3, n_joints), dtype=float)
    for joint in range(body_index + 1):
        joint_frame = np.asarray(frames[joint], dtype=float)
        origin = joint_frame[:3, 3]
        axis = joint_frame[:3, 2]
        if links[joint].joint_type.lower() == "revolute":
            linear[:, joint] = np.cross(axis, position - origin)
            angular[:, joint] = axis
        else:
            linear[:, joint] = axis
    return linear, angular, rotation


def _mass_matrix_single(links, inertias, joint_values, base=None):
    _, frames = forward_kinematics(links, joint_values=joint_values, base=base, return_all=True)
    matrix = np.zeros((len(links), len(links)), dtype=float)
    for body_index, inertia in enumerate(inertias):
        linear, angular, rotation = _body_jacobians(links, frames, body_index, inertia)
        world_inertia = rotation @ np.asarray(inertia.inertia, dtype=float) @ rotation.T
        matrix += inertia.mass * linear.T @ linear + angular.T @ world_inertia @ angular
    return 0.5 * (matrix + matrix.T)


def _mass_and_gravity_single(links, inertias, joint_values, gravity, base=None):
    """Compute coupled mass and gravity terms from one forward-kinematics pass."""

    _, frames = forward_kinematics(links, joint_values=joint_values, base=base, return_all=True)
    matrix = np.zeros((len(links), len(links)), dtype=float)
    forces = np.zeros(len(links), dtype=float)
    for body_index, inertia in enumerate(inertias):
        linear, angular, rotation = _body_jacobians(links, frames, body_index, inertia)
        world_inertia = rotation @ np.asarray(inertia.inertia, dtype=float) @ rotation.T
        matrix += inertia.mass * linear.T @ linear + angular.T @ world_inertia @ angular
        forces -= linear.T @ (inertia.mass * gravity)
    return 0.5 * (matrix + matrix.T), forces


def mass_matrix_batch(links, link_inertias, joint_values, *, base=None, output=None):
    """Return one joint-space mass matrix per configuration.

    The returned shape is ``(n_samples, n_joints, n_joints)``.  ``output``
    can reuse a contiguous float64 array with this shape.
    """

    links, inertias = _normalized_inputs(links, link_inertias)
    values = _joint_batch(joint_values, len(links), "joint_values")
    result = _output_buffer(output, (values.shape[0], len(links), len(links)), "output")
    cpp = _load_cpp_dynamics(get_backend())
    if cpp is not None and callable(getattr(cpp, "mass_matrix_batch_dh", None)):
        cpp.mass_matrix_batch_dh(*_cpp_dynamics_arguments(links, inertias, values), base, result)
        return result
    for index, configuration in enumerate(values):
        result[index] = _mass_matrix_single(links, inertias, configuration, base=base)
    return result


def mass_matrix(links, link_inertias, joint_values=None, *, base=None):
    """Return the joint-space mass matrix at one configuration."""

    links, inertias = _normalized_inputs(links, link_inertias)
    if joint_values is None:
        joint_values = np.zeros(len(links), dtype=float)
    values = np.asarray(joint_values, dtype=float).reshape(-1)
    if values.shape != (len(links),) or not np.all(np.isfinite(values)):
        raise ValueError(f"joint_values must be a finite vector with shape ({len(links)},)")
    return _mass_matrix_single(links, inertias, values, base=base)


def _gravity_single(links, inertias, joint_values, gravity, base=None):
    _, frames = forward_kinematics(links, joint_values=joint_values, base=base, return_all=True)
    forces = np.zeros(len(links), dtype=float)
    for body_index, inertia in enumerate(inertias):
        linear, _, _ = _body_jacobians(links, frames, body_index, inertia)
        forces -= linear.T @ (inertia.mass * gravity)
    return forces


def gravity_forces_batch(links, link_inertias, joint_values, *, gravity=(0.0, 0.0, -9.81), base=None, output=None):
    """Return holding torques caused by gravitational acceleration."""

    links, inertias = _normalized_inputs(links, link_inertias)
    values = _joint_batch(joint_values, len(links), "joint_values")
    gravity = np.asarray(gravity, dtype=float).reshape(-1)
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
        raise ValueError("gravity must be a finite 3-vector")
    result = _output_buffer(output, values.shape, "output")
    cpp = _load_cpp_dynamics(get_backend())
    if cpp is not None and callable(getattr(cpp, "gravity_forces_batch_dh", None)):
        cpp.gravity_forces_batch_dh(
            *_cpp_dynamics_arguments(links, inertias, values), gravity, base, result
        )
        return result
    for index, configuration in enumerate(values):
        result[index] = _gravity_single(links, inertias, configuration, gravity, base=base)
    return result


def gravity_forces(links, link_inertias, joint_values=None, *, gravity=(0.0, 0.0, -9.81), base=None):
    """Return gravity holding torques at one configuration."""

    links, inertias = _normalized_inputs(links, link_inertias)
    if joint_values is None:
        joint_values = np.zeros(len(links), dtype=float)
    values = np.asarray(joint_values, dtype=float).reshape(-1)
    gravity = np.asarray(gravity, dtype=float).reshape(-1)
    if values.shape != (len(links),) or not np.all(np.isfinite(values)):
        raise ValueError(f"joint_values must be a finite vector with shape ({len(links)},)")
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
        raise ValueError("gravity must be a finite 3-vector")
    return _gravity_single(links, inertias, values, gravity, base=base)


def coriolis_forces_batch(
    links, link_inertias, joint_values, joint_velocities, *, base=None, epsilon=1e-6, output=None
):
    """Return Coriolis and centrifugal forces from mass-matrix derivatives.

    The finite-difference formulation provides a reliable reference for the
    serial DH mass model.  The optional C++ backend evaluates the same
    derivatives and Christoffel contraction without Python-level per-joint
    dispatch.
    """

    links, inertias = _normalized_inputs(links, link_inertias)
    values = _joint_batch(joint_values, len(links), "joint_values")
    velocities = _joint_batch(joint_velocities, len(links), "joint_velocities")
    if velocities.shape != values.shape:
        raise ValueError("joint_velocities must have the same shape as joint_values")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a positive finite value")
    samples, joints = values.shape
    result = _output_buffer(output, values.shape, "output")
    cpp = _load_cpp_dynamics(get_backend())
    if cpp is not None and callable(getattr(cpp, "coriolis_forces_batch_dh", None)):
        cpp.coriolis_forces_batch_dh(
            *_cpp_dynamics_arguments(links, inertias, values),
            np.ascontiguousarray(velocities, dtype=float),
            base,
            epsilon,
            result,
        )
        return result
    derivatives = np.empty((samples, joints, joints, joints), dtype=float)
    for coordinate in range(joints):
        delta = np.zeros_like(values)
        delta[:, coordinate] = epsilon
        plus = mass_matrix_batch(links, inertias, values + delta, base=base)
        minus = mass_matrix_batch(links, inertias, values - delta, base=base)
        derivatives[:, coordinate] = (plus - minus) / (2.0 * epsilon)
    result.fill(0.0)
    for row in range(joints):
        for first in range(joints):
            for second in range(joints):
                coefficient = 0.5 * (
                    derivatives[:, second, row, first]
                    + derivatives[:, first, row, second]
                    - derivatives[:, row, first, second]
                )
                result[:, row] += coefficient * velocities[:, first] * velocities[:, second]
    return result


def inverse_dynamics_batch(
    links,
    link_inertias,
    joint_values,
    joint_velocities,
    joint_accelerations,
    *,
    gravity=(0.0, 0.0, -9.81),
    base=None,
    include_coriolis=False,
    output=None,
):
    """Return joint torques for ``M(q) qdd + c(q, qd) + g(q)``."""

    links, inertias = _normalized_inputs(links, link_inertias)
    values = _joint_batch(joint_values, len(links), "joint_values")
    velocities = _joint_batch(joint_velocities, len(links), "joint_velocities")
    accelerations = _joint_batch(joint_accelerations, len(links), "joint_accelerations")
    if velocities.shape != values.shape or accelerations.shape != values.shape:
        raise ValueError("joint velocities and accelerations must match joint_values")
    result = _output_buffer(output, values.shape, "output")
    gravity = np.asarray(gravity, dtype=float).reshape(-1)
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
        raise ValueError("gravity must be a finite 3-vector")
    cpp = _load_cpp_dynamics(get_backend())
    if cpp is not None and callable(getattr(cpp, "inverse_dynamics_batch_dh", None)):
        cpp.inverse_dynamics_batch_dh(
            *_cpp_dynamics_arguments(links, inertias, values),
            np.ascontiguousarray(accelerations, dtype=float),
            gravity,
            base,
            result,
        )
    else:
        for index, configuration in enumerate(values):
            matrix, gravity_term = _mass_and_gravity_single(
                links, inertias, configuration, gravity, base=base
            )
            result[index] = matrix @ accelerations[index] + gravity_term
    if include_coriolis and np.any(velocities):
        result += coriolis_forces_batch(links, inertias, values, velocities, base=base)
    return result


def forward_dynamics_batch(
    links,
    link_inertias,
    joint_values,
    joint_velocities,
    joint_torques,
    *,
    gravity=(0.0, 0.0, -9.81),
    base=None,
    include_coriolis=False,
    output=None,
):
    """Solve batched joint accelerations from generalized torque inputs."""

    links, inertias = _normalized_inputs(links, link_inertias)
    values = _joint_batch(joint_values, len(links), "joint_values")
    velocities = _joint_batch(joint_velocities, len(links), "joint_velocities")
    torques = _joint_batch(joint_torques, len(links), "joint_torques")
    if velocities.shape != values.shape or torques.shape != values.shape:
        raise ValueError("joint velocities and torques must match joint_values")
    result = _output_buffer(output, values.shape, "output")
    gravity = np.asarray(gravity, dtype=float).reshape(-1)
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
        raise ValueError("gravity must be a finite 3-vector")
    matrices = np.empty((values.shape[0], len(links), len(links)), dtype=float)
    rhs = np.empty_like(values)
    cpp = _load_cpp_dynamics(get_backend())
    if cpp is not None and callable(getattr(cpp, "mass_and_gravity_batch_dh", None)):
        cpp.mass_and_gravity_batch_dh(
            *_cpp_dynamics_arguments(links, inertias, values), gravity, base, matrices, rhs
        )
        rhs[:] = torques - rhs
    else:
        for index, configuration in enumerate(values):
            matrices[index], gravity_term = _mass_and_gravity_single(
                links, inertias, configuration, gravity, base=base
            )
            rhs[index] = torques[index] - gravity_term
    if include_coriolis and np.any(velocities):
        rhs -= coriolis_forces_batch(links, inertias, values, velocities, base=base)
    if matrices.shape[1] == 0:
        result.fill(0.0)
    else:
        result[:] = np.linalg.solve(matrices, rhs[..., None])[..., 0]
    return result


__all__ = [
    "LinkInertia",
    "coriolis_forces_batch",
    "forward_dynamics_batch",
    "gravity_forces",
    "gravity_forces_batch",
    "inverse_dynamics_batch",
    "mass_matrix",
    "mass_matrix_batch",
    "normalize_link_inertias",
]
