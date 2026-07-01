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
