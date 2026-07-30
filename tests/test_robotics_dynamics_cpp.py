import numpy as np
import pytest

from mastermlx import get_backend, set_backend
from mastermlx.robotics import DHLink, LinkInertia, RobotModel, robotics_backend_report
from mastermlx.robotics.dynamics import _load_cpp_dynamics


def _robot():
    base = np.array(
        [[0.0, -1.0, 0.0, 0.2], [1.0, 0.0, 0.0, -0.1], [0.0, 0.0, 1.0, 0.3], [0.0, 0.0, 0.0, 1.0]]
    )
    return RobotModel.from_dh(
        [
            DHLink(a=0.35, alpha=0.2, d=0.1, theta=0.05),
            DHLink(a=0.2, alpha=-0.4, d=0.0, theta=0.1, joint_type="prismatic", offset=0.02),
            DHLink(a=0.15, alpha=0.3, d=0.05, theta=-0.2),
        ],
        base=base,
        link_inertias=[
            LinkInertia(1.3, (0.1, -0.02, 0.04), (0.02, 0.03, 0.04)),
            LinkInertia(0.7, (-0.05, 0.01, 0.08), (0.01, 0.02, 0.015)),
            LinkInertia(0.4, (0.03, 0.02, -0.04), (0.005, 0.006, 0.008)),
        ],
    )


def test_cpp_dynamics_report_is_consistent():
    report = robotics_backend_report()
    assert isinstance(report["cpp_dynamics"], bool)


def test_cpp_dynamics_matches_numpy_and_reuses_outputs():
    if _load_cpp_dynamics("auto") is None:
        pytest.skip("C++ dynamics extension is unavailable")

    robot = _robot()
    values = np.array([[0.1, 0.03, -0.2], [-0.4, -0.02, 0.25], [0.7, 0.08, 0.1]])
    velocities = np.array([[0.2, -0.04, 0.1], [-0.3, 0.02, -0.15], [0.05, 0.08, 0.2]])
    accelerations = np.array([[0.5, -0.1, 0.3], [-0.2, 0.04, -0.1], [0.1, 0.2, -0.3]])
    gravity = np.array([0.3, -9.81, 0.5])
    old = get_backend()
    try:
        set_backend("numpy")
        reference_mass = robot.mass_matrix_batch(values)
        reference_gravity = robot.gravity_forces_batch(values, gravity=gravity)
        reference_torque = robot.inverse_dynamics_batch(values, velocities, accelerations, gravity=gravity)
        reference_coriolis_forces = robot.coriolis_forces_batch(values, velocities)
        reference_forward = robot.forward_dynamics_batch(
            values, velocities, reference_torque, gravity=gravity
        )
        reference_coriolis = robot.inverse_dynamics_batch(
            values, velocities, accelerations, gravity=gravity, include_coriolis=True
        )

        set_backend("auto")
        mass_output = np.empty_like(reference_mass)
        gravity_output = np.empty_like(reference_gravity)
        torque_output = np.empty_like(reference_torque)
        coriolis_output = np.empty_like(reference_torque)
        forward_output = np.empty_like(reference_forward)
        mass = robot.mass_matrix_batch(values, output=mass_output)
        forces = robot.gravity_forces_batch(values, gravity=gravity, output=gravity_output)
        torque = robot.inverse_dynamics_batch(
            values, velocities, accelerations, gravity=gravity, output=torque_output
        )
        coriolis = robot.coriolis_forces_batch(values, velocities, output=coriolis_output)
        forward = robot.forward_dynamics_batch(
            values, velocities, torque, gravity=gravity, output=forward_output
        )
        with_coriolis = robot.inverse_dynamics_batch(
            values, velocities, accelerations, gravity=gravity, include_coriolis=True
        )
    finally:
        set_backend(old)

    assert mass is mass_output
    assert forces is gravity_output
    assert torque is torque_output
    assert coriolis is coriolis_output
    assert forward is forward_output
    assert np.allclose(mass, reference_mass, atol=1e-12)
    assert np.allclose(forces, reference_gravity, atol=1e-12)
    assert np.allclose(torque, reference_torque, atol=1e-12)
    assert np.allclose(coriolis, reference_coriolis_forces, atol=1e-10)
    assert np.allclose(forward, reference_forward, atol=1e-12)
    assert np.allclose(with_coriolis, reference_coriolis, atol=1e-10)


def test_cpp_dynamics_parallel_batch_matches_numpy():
    if _load_cpp_dynamics("auto") is None:
        pytest.skip("C++ dynamics extension is unavailable")

    robot = _robot()
    values = np.tile(
        np.array([[0.1, 0.03, -0.2], [-0.4, -0.02, 0.25], [0.7, 0.08, 0.1]]),
        (200, 1),
    )
    old = get_backend()
    try:
        set_backend("numpy")
        reference = robot.mass_matrix_batch(values)
        set_backend("auto")
        accelerated = robot.mass_matrix_batch(values)
    finally:
        set_backend(old)

    assert np.allclose(accelerated, reference, atol=1e-12)
