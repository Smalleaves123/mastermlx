import numpy as np

from mastermlx.robotics import RobotModel
from mastermlx.sim import (
    RobotSimulation,
    SimpleRobotSim,
    SimpleWorld,
    load_world_config,
    step_state,
    step_state_batch,
)


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


def test_robot_model_from_dh():
    robot = RobotModel.from_dh(
        [
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        ],
        name="planar2r-dh",
    )
    assert robot.name == "planar2r-dh"
    assert np.allclose(robot.fk([0.0, 0.0])[:3, 3], np.array([2.0, 0.0, 0.0]))


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


def test_step_state_batch_matches_independent_rollouts():
    robot = RobotModel.from_urdf(_planar_2r_urdf())
    actions = np.array(
        [
            [[0.5, -0.5], [0.0, 0.0], [0.2, 0.2]],
            [[-0.2, 0.3], [0.1, 0.0], [0.0, -0.1]],
        ]
    )
    initial_states = np.array([[0.1, -0.1, 0.0, 0.0], [-0.2, 0.2, 0.1, -0.1]])
    sim = SimpleRobotSim(robot, dt=0.1, damping=0.2)
    batch = sim.rollout_batch(actions, initial_states=initial_states)

    expected = []
    for action_sequence, initial_state in zip(actions, initial_states):
        sim.reset(initial_state)
        states, _ = sim.rollout(action_sequence)
        expected.append(states)

    assert batch.shape == (2, 4, 4)
    assert np.allclose(batch, np.asarray(expected))
    assert np.allclose(
        step_state_batch(initial_states, actions[:, 0], dt=0.1, damping=0.2),
        batch[:, 1],
    )


def test_robot_simulation_protocol_and_grasp_release():
    robot = RobotModel.from_urdf(_planar_2r_urdf())
    world = SimpleWorld(robot)
    item = world.add_object("part", [2.0, 0.0, 0.0], radius=0.05)
    env = RobotSimulation(world, dt=0.1, max_steps=4, grasp_distance=0.2)

    observation, reset_info = env.reset(seed=7)
    assert observation["state"].shape == (4,)
    assert observation["object_positions"].shape == (1, 3)
    assert reset_info["seed"] == 7

    observation, reward, terminated, truncated, info = env.step(
        {"joint": [0.0, 0.0], "gripper": 0.0}
    )
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated
    assert info["event"] == {"command": "grasp", "object": "part"}
    assert item.attached
    assert observation["object_attached"].tolist() == [True]

    _, _, terminated, truncated, info = env.step({"joint": [0.0, 0.0], "gripper": 1.0})
    assert not terminated
    assert not truncated
    assert info["event"] == {"command": "release", "object": "part"}
    assert not item.attached


def test_robot_simulation_target_and_object_config():
    cfg = {
        "robot": {
            "links": [
                {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
                {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            ]
        },
        "objects": [{"name": "box", "position": [2.0, 0.0], "radius": 0.1}],
    }
    world, _, _ = load_world_config(cfg)
    env = RobotSimulation(world, target_position=[2.0, 0.0, 0.0], max_steps=2)
    _, _, terminated, truncated, info = env.step([0.0, 0.0])

    assert world.objects[0].position.shape == (3,)
    assert terminated
    assert not truncated
    assert info["success"]
