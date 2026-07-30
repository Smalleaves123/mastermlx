import numpy as np

from mastermlx.robotics import (
    BoxObstacle,
    CapsuleObstacle,
    RobotModel,
    SphereObstacle,
    path_collision_free,
    path_collision_report,
    path_collision_summary,
    robot_collision_report,
    segment_distance,
)
from mastermlx.sim import SimpleWorld, load_world_config


def _robot():
    return RobotModel.from_dh(
        [
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        ],
        name="collision-planar2r",
    )


def test_collision_obstacles_report_link_hits_and_clearance():
    robot = _robot()
    obstacles = [
        SphereObstacle((1.0, 0.0), 0.1),
        BoxObstacle((1.4, -0.1), (1.6, 0.1)),
        CapsuleObstacle((0.0, 0.4), (2.0, 0.4), 0.05),
    ]

    report = robot_collision_report(robot, [0.0, 0.0], obstacles)

    assert report["collision"]
    assert report["minimum_clearance"] < 0.0
    assert any(hit["kind"] == "point" for hit in report["hits"])
    assert any(hit["kind"] == "segment" for hit in report["hits"])


def test_path_collision_report_interpolates_joint_edges():
    robot = _robot()
    obstacle = SphereObstacle((1.5, 0.0), 0.15)
    path = np.array([[np.pi / 2.0, 0.0], [-np.pi / 2.0, 0.0]])

    report = path_collision_report(robot, path, [obstacle], interpolation_step=0.05)

    assert report["collision"]
    assert report["first_collision_index"] is not None
    assert report["n_samples"] > path.shape[0]


def test_path_collision_summary_matches_detailed_report_arrays():
    robot = _robot()
    obstacle = SphereObstacle((1.5, 0.0), 0.15)
    path = np.array([[np.pi / 2.0, 0.0], [-np.pi / 2.0, 0.0]])

    detailed = path_collision_report(robot, path, [obstacle], interpolation_step=0.05)
    summary = path_collision_summary(robot, path, [obstacle], interpolation_step=0.05)

    assert summary["collision"] == detailed["collision"]
    assert summary["first_collision_index"] == detailed["first_collision_index"]
    assert summary["n_samples"] == detailed["n_samples"]
    assert np.allclose(summary["samples"], detailed["samples"])
    assert np.allclose(summary["clearances"], detailed["clearances"])
    assert np.isclose(summary["minimum_clearance"], detailed["minimum_clearance"])
    assert not path_collision_free(robot, path, [obstacle], interpolation_step=0.05)


def test_segment_distance_and_world_config_support_new_obstacles():
    assert segment_distance([0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.5, 1.0]) == 0.5

    world, state, sim_cfg = load_world_config(
        {
            "robot": {
                "links": [
                    {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
                    {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
                ]
            },
            "obstacles": [
                {"kind": "box", "lower": [0.8, -0.1], "upper": [1.2, 0.1]},
                {"kind": "capsule", "start": [0.0, 0.5], "end": [2.0, 0.5], "radius": 0.05},
            ],
            "state": [0.0, 0.0, 0.0, 0.0],
            "sim": {"dt": 0.05},
        }
    )

    assert isinstance(world, SimpleWorld)
    assert state.shape == (4,)
    assert sim_cfg["dt"] == 0.05
    assert world.hit([0.0, 0.0])
