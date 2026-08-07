import numpy as np

from mastermlx.robotics import (
    RobotModel,
    RobotWorkcell,
    SphereObstacle,
    camera_visibility_matrix,
    evaluate_inspection_coverage,
    generate_viewpoint_candidates,
    look_at_pose,
    select_inspection_viewpoints,
)
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


def test_camera_visibility_applies_rectangular_frustum_and_ray_occlusion():
    pose = look_at_pose([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    visible = camera_visibility_matrix(
        [pose],
        points,
        coverage_radius=2.0,
        horizontal_field_of_view=60.0,
        vertical_field_of_view=60.0,
    )

    assert visible["visible"].tolist() == [[True, False]]

    blocked = camera_visibility_matrix(
        [pose],
        points[:1],
        [True],
        [SphereObstacle((0.0, 0.0, 0.5), 0.1)],
        coverage_radius=2.0,
        horizontal_field_of_view=60.0,
        vertical_field_of_view=60.0,
    )
    assert blocked["occluded"].tolist() == [[True]]


def test_greedy_viewpoint_selection_skips_unreachable_full_coverage_view():
    poses = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
    ]
    visibility = np.array(
        [
            [True, True, True, True],
            [True, True, False, False],
            [False, False, True, True],
        ]
    )
    result = select_inspection_viewpoints(
        visibility,
        poses,
        reachable=[False, True, True],
        required_coverage=1.0,
        start_position=[0.0, 0.0, 0.0],
        travel_speed=2.0,
        dwell_time=0.1,
    )

    assert result["selected_indices"].tolist() == [1, 2]
    assert result["coverage_rate"] == 1.0
    assert result["required_coverage_met"]
    assert result["estimated_total_time"] > 0.0


def test_active_inspection_selects_candidates_then_plans_tcp_route():
    robot, workcell = _inspection_workcell()
    configurations = np.array([[0.0, -0.4], [0.2, -0.3], [0.4, -0.2]])
    candidates = [pose[:3, 3] for pose in robot.fk_batch(configurations)]
    points = np.asarray(candidates) + np.array([0.0, 0.04, 0.0])

    result = workcell.plan_active_inspection(
        points,
        [0.0, -0.4],
        candidate_poses=candidates,
        coverage_radius=0.08,
        dwell_time=0.05,
        velocity_limits=0.8,
    )
    report = result["active_inspection_report"]

    assert report["n_candidate_poses"] == 3
    assert report["n_selected_views"] == 3
    assert report["selection_coverage_rate"] == 1.0
    assert result["candidate_visibility"]["visible"].shape == (3, 3)
    assert report["execution_ready"]


def test_generate_viewpoint_candidates_produces_look_at_poses():
    candidates = generate_viewpoint_candidates(
        [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
        standoff=0.5,
        directions=[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    )

    assert len(candidates) == 2
    assert all(candidate.shape == (4, 4) for candidate in candidates)
    assert np.allclose(candidates[0][:3, :3] @ candidates[0][:3, :3].T, np.eye(3))
