import numpy as np

from mastermlx.robotics import RobotModel
from mastermlx.sim import SimpleRobotSim, step_state


def _planar_2r_urdf():
    return """
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


def test_robot_model_fk_jacobian_and_ik():
    robot = RobotModel.from_urdf(_planar_2r_urdf(), name="planar2r")
    T = robot.fk([0.0, 0.0])
    J = robot.jacobian([0.0, 0.0])
    q = robot.ik(np.array([2.0, 0.0, 0.0]), joint_values=[0.1, -0.1], max_iter=200)

    assert robot.name == "planar2r"
    assert np.allclose(T[:3, 3], np.array([2.0, 0.0, 0.0]))
    assert J.shape == (6, 2)
    assert np.all(np.isfinite(J))
    assert np.allclose(robot.fk(q)[:3, 3], np.array([2.0, 0.0, 0.0]), atol=1e-4)


def test_simple_robot_sim_steps_state():
    robot = RobotModel.from_urdf(_planar_2r_urdf())
    sim = SimpleRobotSim(robot, dt=0.1, damping=0.2)
    next_state = sim.step(np.array([1.0, -1.0]))
    raw_next = step_state(np.zeros(4), np.array([1.0, -1.0]), dt=0.1, damping=0.2)

    assert next_state.shape == (4,)
    assert np.allclose(next_state, raw_next)
    assert np.allclose(sim.pose()[:3, 3], robot.fk(sim.q)[:3, 3])


def test_simple_robot_sim_rollout_and_render():
    robot = RobotModel.from_urdf(_planar_2r_urdf())
    sim = SimpleRobotSim(robot, dt=0.1, damping=0.2)
    actions = np.array([[0.5, -0.5], [0.0, 0.0], [0.2, 0.2]])
    states, poses = sim.rollout(actions)

    assert states.shape == (4, 4)
    assert len(poses) == 4
    assert all(p.shape == (4, 4) for p in poses)
    ax = sim.render()
    assert ax is not None
