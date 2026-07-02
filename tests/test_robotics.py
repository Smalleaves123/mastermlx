import numpy as np

from mastermlx.robotics import (
    DHLink,
    cubic_time_scaling,
    invert_transform,
    forward_kinematics,
    geometric_jacobian,
    inverse_kinematics,
    joint_trajectory,
    plan_joint_path,
    plan_joint_trajectory,
    parse_urdf,
    matrix_to_euler,
    matrix_to_quaternion,
    planar_2r_jacobian,
    smooth_joint_path,
    quaternion_to_matrix,
    quintic_time_scaling,
    sample_joint_trajectory,
    sample_joint_trajectory_segments,
    urdf_to_dh_chain,
    transform_points,
    rot_z,
)


def test_rotation_quaternion_round_trip():
    R = rot_z(np.pi / 2)
    q = matrix_to_quaternion(R)
    R2 = quaternion_to_matrix(q)
    assert np.allclose(R, R2)


def test_rotation_euler_round_trip():
    R = rot_z(0.3) @ rot_z(0.0)
    e = matrix_to_euler(R)
    assert np.allclose(e[1], 0.0, atol=1e-12)
    assert np.allclose(rot_z(e[2]), R, atol=1e-12)


def test_transform_points_and_inverse_transform():
    T = np.array([
        [0.0, -1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = transform_points(T, pts)
    inv = invert_transform(T)
    back = transform_points(inv, out)
    assert np.allclose(out, np.array([[1.0, 3.0, 3.0], [0.0, 2.0, 3.0]]))
    assert np.allclose(back, pts)


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


def test_sample_joint_trajectory_segments_are_continuous():
    waypoints = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 0.0],
    ])
    times, positions, velocities, accelerations = sample_joint_trajectory_segments(
        waypoints,
        durations=[1.0, 2.0],
        num_samples_per_segment=4,
        kind="quintic",
    )

    assert times.shape == (7,)
    assert positions.shape == (7, 2)
    assert velocities.shape == (7, 2)
    assert accelerations.shape == (7, 2)
    assert np.allclose(positions[0], waypoints[0])
    assert np.allclose(positions[-1], waypoints[-1])
    assert np.allclose(positions[3], waypoints[1])
    assert np.allclose(times[3], 1.0)
    assert np.allclose(times[4], 1.66666667, atol=1e-8)


def test_plan_joint_path_and_smoothing_reduce_kinks():
    q0 = np.array([0.0, 0.0])
    qf = np.array([2.0, 1.0])
    via = [np.array([0.8, 0.9]), np.array([1.2, 0.2])]

    path = plan_joint_path(q0, qf, via_points=via)
    smoothed = smooth_joint_path(path, smoothness=1.0)

    assert np.allclose(path[0], q0)
    assert np.allclose(path[-1], qf)
    assert np.allclose(smoothed[0], q0)
    assert np.allclose(smoothed[-1], qf)

    raw_kink = np.linalg.norm(path[2] - 2.0 * path[1] + path[0])
    smooth_kink = np.linalg.norm(smoothed[2] - 2.0 * smoothed[1] + smoothed[0])
    assert smooth_kink <= raw_kink


def test_plan_joint_trajectory_returns_samples():
    times, positions, velocities, accelerations = plan_joint_trajectory(
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        duration=2.0,
        num_waypoints=5,
        num_samples_per_segment=4,
        smoothness=1.0,
    )

    assert times.ndim == 1
    assert positions.shape[1] == 2
    assert velocities.shape == positions.shape
    assert accelerations.shape == positions.shape
    assert np.allclose(positions[0], np.array([0.0, 0.0]))
    assert np.allclose(positions[-1], np.array([1.0, 1.0]))


def test_inverse_kinematics_reaches_planar_target():
    links = [
        DHLink(a=1.0, alpha=0.0, d=0.0, theta=0.0),
        DHLink(a=1.0, alpha=0.0, d=0.0, theta=0.0),
    ]
    q = inverse_kinematics(np.array([2.0, 0.0, 0.0]), links, joint_values=[0.1, -0.1], max_iter=200)
    T = forward_kinematics(links, q)
    assert np.allclose(T[:3, 3], np.array([2.0, 0.0, 0.0]), atol=1e-4)


def test_urdf_parser_and_chain_conversion():
    xml = """
    <robot name="planar2r">
      <link name="base" />
      <link name="link1" />
      <link name="link2" />
      <joint name="joint1" type="revolute">
        <parent link="base" />
        <child link="link1" />
        <origin xyz="1 0 0" rpy="0 0 0" />
        <axis xyz="0 0 1" />
      </joint>
      <joint name="joint2" type="revolute">
        <parent link="link1" />
        <child link="link2" />
        <origin xyz="1 0 0" rpy="0 0 0" />
        <axis xyz="0 0 1" />
      </joint>
    </robot>
    """
    links, joints = parse_urdf(xml)
    chain = urdf_to_dh_chain(xml)
    assert len(links) == 3
    assert len(joints) == 2
    assert len(chain) == 2
    assert chain[0].a == 1.0
    assert chain[1].a == 1.0
