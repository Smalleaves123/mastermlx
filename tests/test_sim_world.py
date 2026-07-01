import numpy as np

from mastermlx.robotics import RobotModel
from mastermlx.sim import SimpleWorld


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
