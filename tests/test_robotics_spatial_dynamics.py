import numpy as np

from mastermlx import get_backend, set_backend
from mastermlx.robotics import URDFRobotModel
from mastermlx.robotics.urdf_parser import _load_cpp_spatial


def _dynamic_urdf():
    return """
    <robot name="dynamic_arm">
      <link name="base" />
      <link name="link1">
        <inertial>
          <origin xyz="0.5 0 0" rpy="0 0 0" />
          <mass value="1.0" />
          <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02" />
        </inertial>
      </link>
      <link name="link2">
        <inertial>
          <origin xyz="0.4 0 0" rpy="0 0 0" />
          <mass value="0.8" />
          <inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.015" />
        </inertial>
      </link>
      <joint name="joint1" type="revolute">
        <parent link="base" /><child link="link1" />
        <origin xyz="0 0 0" rpy="0 0 0" />
        <axis xyz="0 0 1" /><limit lower="-3.14" upper="3.14" />
      </joint>
      <joint name="joint2" type="revolute">
        <parent link="link1" /><child link="link2" />
        <origin xyz="1 0 0" rpy="0 0 0" />
        <axis xyz="0 0 1" /><limit lower="-3.14" upper="3.14" />
      </joint>
    </robot>
    """


def _mixed_dynamic_urdf():
    return """
    <robot name="mixed_dynamic_arm">
      <link name="base" />
      <link name="fixed_link">
        <inertial><origin xyz="0.1 0 0" rpy="0.1 0.2 0.3" />
          <mass value="0.7" />
          <inertia ixx="0.02" ixy="0.001" ixz="0" iyy="0.025" iyz="0.001" izz="0.03" />
        </inertial>
      </link>
      <link name="slider">
        <inertial><origin xyz="0 0.1 0" rpy="0 0 0" />
          <mass value="0.9" />
          <inertia ixx="0.015" ixy="0" ixz="0.001" iyy="0.02" iyz="0" izz="0.025" />
        </inertial>
      </link>
      <link name="tip">
        <inertial><origin xyz="0.15 0 0.02" rpy="0 0 0" />
          <mass value="0.6" />
          <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.012" iyz="0.001" izz="0.016" />
        </inertial>
      </link>
      <joint name="fixed" type="fixed">
        <parent link="base" /><child link="fixed_link" />
        <origin xyz="0.2 -0.1 0.3" rpy="0.2 -0.1 0.15" />
      </joint>
      <joint name="slide" type="prismatic">
        <parent link="fixed_link" /><child link="slider" />
        <origin xyz="0.3 0.05 0" rpy="-0.1 0.25 -0.2" />
        <axis xyz="1 2 1" /><limit lower="-0.4" upper="0.4" />
      </joint>
      <joint name="elbow" type="revolute">
        <parent link="slider" /><child link="tip" />
        <origin xyz="0.25 0 0.1" rpy="0.1 0.2 0.05" />
        <axis xyz="1 -1 2" /><limit lower="-1.2" upper="1.2" />
      </joint>
    </robot>
    """


def _rotated_base():
    return np.array(
        [
            [0.0, -1.0, 0.0, 0.3],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.4],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def test_spatial_dynamics_terms_and_forward_inverse_consistency():
    robot = URDFRobotModel.from_urdf(_dynamic_urdf())
    q = np.array([0.3, -0.4])
    qd = np.array([0.2, -0.1])
    qdd = np.array([0.4, -0.25])

    mass = robot.mass_matrix(q)
    gravity = robot.gravity_forces(q)
    coriolis = robot.coriolis_forces(q, qd)
    torque = robot.inverse_dynamics(q, qd, qdd)
    recovered = robot.forward_dynamics(q, qd, torque)

    assert mass.shape == (2, 2)
    assert np.allclose(mass, mass.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(mass) > 0.0)
    assert gravity.shape == (2,)
    assert coriolis.shape == (2,)
    assert np.allclose(recovered, qdd, atol=2e-6)
    assert np.allclose(torque, mass @ qdd + coriolis + gravity, atol=2e-6)


def test_cpp_spatial_dynamics_matches_numpy():
    if _load_cpp_spatial("auto") is None:
        return
    robot = URDFRobotModel.from_urdf(_dynamic_urdf())
    q = np.array([0.3, -0.4])
    qd = np.array([0.2, -0.1])
    qdd = np.array([0.4, -0.25])
    torques = np.array([0.2, -0.1])
    old = get_backend()
    try:
        set_backend("numpy")
        reference = (
            robot.mass_matrix(q),
            robot.gravity_forces(q),
            robot.coriolis_forces(q, qd),
            robot.inverse_dynamics(q, qd, qdd),
            robot.forward_dynamics(q, qd, torques),
        )
        set_backend("auto")
        accelerated = (
            robot.mass_matrix(q),
            robot.gravity_forces(q),
            robot.coriolis_forces(q, qd),
            robot.inverse_dynamics(q, qd, qdd),
            robot.forward_dynamics(q, qd, torques),
        )
    finally:
        set_backend(old)

    for actual, expected in zip(accelerated, reference):
        assert np.allclose(actual, expected, atol=2e-10)


def test_cpp_spatial_dynamics_batch_matches_numpy():
    if _load_cpp_spatial("auto") is None:
        return
    robot = URDFRobotModel.from_urdf(_dynamic_urdf())
    values = np.array([[0.3, -0.4], [-0.2, 0.5], [0.1, 0.2]])
    velocities = np.array([[0.2, -0.1], [0.1, 0.3], [-0.2, 0.15]])
    accelerations = np.array([[0.4, -0.25], [0.2, 0.1], [-0.1, 0.3]])
    torques = np.array([[0.2, -0.1], [0.1, 0.2], [-0.3, 0.15]])
    old = get_backend()
    try:
        set_backend("numpy")
        reference = (
            robot.mass_matrix_batch(values),
            robot.gravity_forces_batch(values),
            robot.coriolis_forces_batch(values, velocities),
            robot.inverse_dynamics_batch(values, velocities, accelerations),
            robot.forward_dynamics_batch(values, velocities, torques),
        )
        set_backend("auto")
        accelerated = (
            robot.mass_matrix_batch(values),
            robot.gravity_forces_batch(values),
            robot.coriolis_forces_batch(values, velocities),
            robot.inverse_dynamics_batch(values, velocities, accelerations),
            robot.forward_dynamics_batch(values, velocities, torques),
        )
    finally:
        set_backend(old)

    for actual, expected in zip(accelerated, reference):
        assert np.allclose(actual, expected, atol=2e-10)


def test_cpp_rnea_matches_numpy_for_fixed_prismatic_and_rotated_base():
    if _load_cpp_spatial("auto") is None:
        return
    robot = URDFRobotModel.from_urdf(_mixed_dynamic_urdf(), base=_rotated_base())
    values = np.array([[0.12, -0.35], [-0.2, 0.4]])
    velocities = np.array([[0.3, -0.25], [-0.15, 0.2]])
    accelerations = np.array([[0.4, -0.1], [-0.2, 0.35]])
    old = get_backend()
    try:
        set_backend("numpy")
        reference = (
            robot.mass_matrix_batch(values),
            robot.gravity_forces_batch(values),
            robot.coriolis_forces_batch(values, velocities),
            robot.inverse_dynamics_batch(values, velocities, accelerations),
            robot.forward_dynamics_batch(values, velocities, np.array([[0.2, -0.1], [0.1, 0.3]])),
        )
        set_backend("auto")
        accelerated = (
            robot.mass_matrix_batch(values),
            robot.gravity_forces_batch(values),
            robot.coriolis_forces_batch(values, velocities),
            robot.inverse_dynamics_batch(values, velocities, accelerations),
            robot.forward_dynamics_batch(values, velocities, np.array([[0.2, -0.1], [0.1, 0.3]])),
        )
    finally:
        set_backend(old)

    for actual, expected in zip(accelerated, reference):
        assert np.allclose(actual, expected, atol=2e-9)


def test_spatial_batch_all_fixed_chain_has_empty_joint_outputs():
    xml = """
    <robot name="fixed_chain">
      <link name="base" />
      <link name="tip">
        <inertial><mass value="1.0" />
          <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0" />
        </inertial>
      </link>
      <joint name="fixed" type="fixed">
        <parent link="base" /><child link="tip" />
        <origin xyz="0.2 0 0.1" rpy="0.1 0 0" />
      </joint>
    </robot>
    """
    robot = URDFRobotModel.from_urdf(xml)
    values = np.empty((2, 0))
    velocities = np.empty((2, 0))
    torques = np.empty((2, 0))

    assert robot.mass_matrix_batch(values).shape == (2, 0, 0)
    assert robot.gravity_forces_batch(values).shape == (2, 0)
    assert robot.coriolis_forces_batch(values, velocities).shape == (2, 0)
    assert robot.inverse_dynamics_batch(values, velocities, torques).shape == (2, 0)
    assert robot.forward_dynamics_batch(values, velocities, torques).shape == (2, 0)


def test_batch_velocities_are_not_checked_against_position_limits():
    robot = URDFRobotModel.from_urdf(_dynamic_urdf())
    values = np.array([[0.1, -0.2]])
    velocities = np.array([[10.0, -12.0]])

    result = robot.coriolis_forces_batch(values, velocities)

    assert result.shape == (1, 2)


def test_spatial_computed_torque_respects_torque_limits():
    robot = URDFRobotModel.from_urdf(_dynamic_urdf())
    command = robot.computed_torque_control(
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, -1.0],
        kp=50.0,
        kd=10.0,
        torque_limits=0.5,
    )

    assert command.shape == (2,)
    assert np.all(np.abs(command) <= 0.5 + 1e-12)
