import numpy as np
import json

from mastermlx.robotics import RobotModel
from mastermlx.sim import SimpleWorld, load_world_config


def _planar_2r_dh():
    return [
        {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
    ]


def test_simple_world_collision_and_scan():
    robot = RobotModel.from_dh(_planar_2r_dh())
    world = SimpleWorld(robot)
    world.add_obstacle(center=(2.0, 0.0), radius=0.25)

    hits = world.collision_report([0.0, 0.0])
    angles, ranges = world.lidar_scan([0.0, 0.0], num_rays=32, max_range=5.0)
    ax = world.render([0.0, 0.0])

    assert len(hits) >= 1
    assert angles.shape == (32,)
    assert ranges.shape == (32,)
    assert np.all(ranges <= 5.0)
    assert ax is not None


def test_world_hit_checks_link_segments():
    robot = RobotModel.from_dh(_planar_2r_dh())
    world = SimpleWorld(robot)
    world.add_obstacle(center=(0.5, 0.0), radius=0.1)

    assert world.hit([0.0, 0.0])
    assert any("segment_index" in item for item in world.collision_report([0.0, 0.0]))
    assert not world.hit([np.pi / 2.0, 0.0])


def test_world_plans_collision_free_joint_path():
    robot = RobotModel.from_dh(_planar_2r_dh())
    world = SimpleWorld(robot)
    world.add_obstacle(center=(1.5, 0.0), radius=0.15)

    path = world.plan_path(
        [np.pi / 2.0, 0.0],
        [-np.pi / 2.0, 0.0],
        bounds=[[-np.pi, np.pi], [-np.pi, np.pi]],
        step=0.15,
        goal_rate=0.25,
        max_iter=3000,
        random_state=0,
    )

    assert path is not None
    assert np.allclose(path[0], [np.pi / 2.0, 0.0])
    assert np.allclose(path[-1], [-np.pi / 2.0, 0.0])
    assert not np.any([world.hit(q) for q in path])


def test_load_world_config_from_dict_and_json():
    cfg = {
        "robot": {
            "name": "planar2r",
            "links": _planar_2r_dh(),
        },
        "obstacles": [{"center": [2.0, 0.0], "radius": 0.25}],
        "state": [0.0, 0.0, 0.0, 0.0],
        "sim": {"dt": 0.05, "damping": 0.1},
    }
    world, state, sim_cfg = load_world_config(cfg)
    world2, state2, sim_cfg2 = load_world_config(json.dumps(cfg))

    assert world.robot.name == "planar2r"
    assert state.shape == (4,)
    assert sim_cfg["dt"] == 0.05
    assert world2.robot.name == "planar2r"
    assert np.allclose(state2, state)
    assert sim_cfg2["damping"] == 0.1


def test_world_trajectory_follow():
    robot = RobotModel.from_dh(_planar_2r_dh())
    world = SimpleWorld(robot)
    trajectory = np.array([
        [0.0, 0.0],
        [0.2, 0.1],
        [0.4, 0.2],
    ])
    states, poses, controls = world.trajectory_follow(trajectory, gains=(2.0, 0.1), dt=0.05)

    assert states.shape == (4, 4)
    assert len(poses) == 4
    assert controls.shape == (3, 2)
    assert np.all(np.isfinite(states))
    assert all(p.shape == (4, 4) for p in poses)
