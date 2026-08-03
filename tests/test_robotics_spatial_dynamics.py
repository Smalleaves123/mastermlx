import numpy as np

from mastermlx.robotics import URDFRobotModel


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
