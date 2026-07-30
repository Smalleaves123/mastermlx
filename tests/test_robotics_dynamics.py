import numpy as np
import pytest

from mastermlx.robotics import LinkInertia, RobotModel


def _pendulum():
    return RobotModel.from_dh(
        [{"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0}],
        name="dynamics-pendulum",
        link_inertias=[
            LinkInertia(
                mass=1.0,
                center_of_mass=(-0.5, 0.0, 0.0),
                inertia=(0.0, 0.0, 0.1),
            )
        ],
    )


def test_link_inertia_and_mass_matrix_match_planar_pendulum():
    robot = _pendulum()
    matrix = robot.mass_matrix([0.0])

    assert matrix.shape == (1, 1)
    assert np.allclose(matrix, [[0.35]])
    assert np.all(np.linalg.eigvalsh(matrix) > 0.0)


def test_batch_dynamics_support_reusable_buffers_and_gravity_torque():
    robot = _pendulum()
    values = np.array([[0.0], [np.pi / 2.0]])
    mass_output = np.empty((2, 1, 1), dtype=float)
    gravity_output = np.empty((2, 1), dtype=float)

    matrices = robot.mass_matrix_batch(values, output=mass_output)
    gravity = robot.gravity_forces_batch(
        values, gravity=(0.0, -9.81, 0.0), output=gravity_output
    )

    assert matrices is mass_output
    assert gravity is gravity_output
    assert np.allclose(matrices[:, 0, 0], 0.35)
    assert np.allclose(gravity[:, 0], [4.905, 0.0], atol=1e-12)


def test_inverse_and_forward_dynamics_round_trip_without_coriolis():
    robot = _pendulum()
    values = np.array([[0.2], [-0.4]])
    velocities = np.array([[0.4], [-0.3]])
    accelerations = np.array([[2.0], [-1.5]])
    torque_output = np.empty_like(values)
    acceleration_output = np.empty_like(values)

    torques = robot.inverse_dynamics_batch(
        values,
        velocities,
        accelerations,
        gravity=(0.0, 0.0, -9.81),
        include_coriolis=True,
        output=torque_output,
    )
    restored = robot.forward_dynamics_batch(
        values,
        velocities,
        torques,
        gravity=(0.0, 0.0, -9.81),
        include_coriolis=True,
        output=acceleration_output,
    )
    coriolis = robot.coriolis_forces_batch(values, velocities)

    assert torques is torque_output
    assert restored is acceleration_output
    assert np.allclose(torques[:, 0], 0.35 * accelerations[:, 0])
    assert np.allclose(restored, accelerations)
    assert np.allclose(coriolis, 0.0)


def test_dynamics_requires_link_mass_properties():
    robot = RobotModel.from_dh([{"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0}])

    with pytest.raises(ValueError, match="link_inertias"):
        robot.mass_matrix([0.0])
