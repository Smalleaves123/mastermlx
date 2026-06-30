import numpy as np

from mastermlx.robotics import (
    DHLink,
    cubic_time_scaling,
    forward_kinematics,
    geometric_jacobian,
    inverse_kinematics,
    joint_trajectory,
    matrix_to_quaternion,
    planar_2r_jacobian,
    quaternion_to_matrix,
    quintic_time_scaling,
    sample_joint_trajectory,
    rot_z,
)


def test_rotation_quaternion_round_trip():
    R = rot_z(np.pi / 2)
    q = matrix_to_quaternion(R)
    R2 = quaternion_to_matrix(q)
    assert np.allclose(R, R2)


def test_forward_kinematics_planar_2r():
    links = [
        DHLink(a=1.0, alpha=0.0, d=0.0, theta=0.0),
        DHLink(a=1.0, alpha=0.0, d=0.0, theta=0.0),
    ]
    T = forward_kinematics(links, [0.0, 0.0])
    assert np.allclose(T[:3, 3], np.array([2.0, 0.0, 0.0]))


def test_geometric_jacobian_matches_planar_formula():
    links = [
        DHLink(a=1.0, alpha=0.0, d=0.0, theta=0.0),
        DHLink(a=1.0, alpha=0.0, d=0.0, theta=0.0),
    ]
    q = np.array([0.3, -0.2])
    J = geometric_jacobian(links, q)
    J2 = planar_2r_jacobian(1.0, 1.0, q[0], q[1])
    assert np.allclose(J[:2, :], J2)


def test_joint_trajectory_endpoints():
    q0 = np.array([0.0, 1.0])
    qf = np.array([1.0, 3.0])
    q_start, qd_start, qdd_start = joint_trajectory(q0, qf, 2.0, 0.0, kind="quintic")
    q_end, qd_end, qdd_end = joint_trajectory(q0, qf, 2.0, 2.0, kind="quintic")

    assert np.allclose(q_start, q0)
    assert np.allclose(q_end, qf)
    assert np.allclose(qd_start, 0.0)
    assert np.allclose(qd_end, 0.0)
    assert np.allclose(qdd_start, 0.0)
    assert np.allclose(qdd_end, 0.0)


def test_time_scaling_bounds():
    s_cubic, ds_cubic, dds_cubic = cubic_time_scaling(1.0, 0.0)
    s_quintic, ds_quintic, dds_quintic = quintic_time_scaling(1.0, 1.0)

    assert np.isclose(s_cubic, 0.0)
    assert np.isclose(ds_cubic, 0.0)
    assert np.isclose(s_quintic, 1.0)
    assert np.isclose(ds_quintic, 0.0)
    assert np.isfinite(dds_cubic)
    assert np.isfinite(dds_quintic)


def test_sample_joint_trajectory_shapes():
    q0 = np.array([0.0, 1.0, -1.0])
    qf = np.array([1.0, 2.0, 0.0])
    times, positions, velocities, accelerations = sample_joint_trajectory(q0, qf, 2.0, num_samples=5, kind="cubic")

    assert times.shape == (5,)
    assert positions.shape == (5, 3)
    assert velocities.shape == (5, 3)
    assert accelerations.shape == (5, 3)
    assert np.allclose(positions[0], q0)
    assert np.allclose(positions[-1], qf)


def test_inverse_kinematics_reaches_planar_target():
    links = [
        DHLink(a=1.0, alpha=0.0, d=0.0, theta=0.0),
        DHLink(a=1.0, alpha=0.0, d=0.0, theta=0.0),
    ]
    q = inverse_kinematics(np.array([2.0, 0.0, 0.0]), links, joint_values=[0.1, -0.1], max_iter=200)
    T = forward_kinematics(links, q)
    assert np.allclose(T[:3, 3], np.array([2.0, 0.0, 0.0]), atol=1e-4)
