import numpy as np

from mastermlx.robotics import RobotModel, RobotWorkcell, SphereObstacle, evaluate_inspection_coverage
from mastermlx.sim import SimpleWorld


def _inspection_workcell():
    robot = RobotModel.from_dh(
        [
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        ],
        name="inspection-planar2r",
    )
    return robot, RobotWorkcell(robot, SimpleWorld(robot))


def test_plan_inspection_reports_coverage_reachability_and_time():
    robot, workcell = _inspection_workcell()
    configurations = np.array([[0.0, -0.4], [0.2, -0.3], [0.4, -0.2]])
    scan_poses = robot.fk_batch(configurations)[:, :3, 3]

    result = workcell.plan_inspection(
        scan_poses,
        [0.0, 0.0],
        inspection_points=scan_poses,
        coverage_radius=0.05,
        dwell_time=0.1,
        velocity_limits=0.8,
    )
    report = result["inspection_report"]

    assert report["execution_ready"]
    assert report["reachability_rate"] == 1.0
    assert report["coverage_rate"] == 1.0
    assert report["occlusion_rate"] == 0.0
    assert report["total_inspection_time"] > report["scan_duration"]
    assert result["scan_waypoint_times"].shape == (3,)


def test_inspection_reports_unreachable_scan_pose_without_raising():
    robot, workcell = _inspection_workcell()
    scan_poses = np.asarray([
        robot.fk([0.0, -0.3])[:3, 3],
        [5.0, 5.0, 0.0],
    ])

    result = workcell.plan_inspection(
        scan_poses,
        [0.0, 0.0],
        inspection_points=scan_poses,
        coverage_radius=0.05,
        check_collisions=False,
    )
    report = result["inspection_report"]

    assert not report["execution_ready"]
    assert report["reachability_rate"] == 0.5
    assert report["first_unreachable_scan_index"] == 1
    assert "unreachable_scan_pose" in report["violations"]
    assert result["joint_targets"].shape == (1, 2)


def test_inspection_coverage_reports_obstacle_occlusion():
    scan_poses = [np.array([2.0, 0.0, 0.0])]
    points = np.array([[0.0, 0.0, 0.0]])
    result = evaluate_inspection_coverage(
        scan_poses,
        points,
        [True],
        [SphereObstacle((1.0, 0.0, 0.0), 0.2)],
        coverage_radius=3.0,
    )

    assert not result["covered"][0]
    assert result["occluded"][0]
    assert result["coverage_rate"] == 0.0
    assert result["occlusion_rate"] == 1.0
