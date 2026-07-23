import csv
import json

import numpy as np
import pytest

from mastermlx.robotics import RobotModel, RobotWorkcell
from mastermlx.sim import SimpleWorld


def _workcell(with_obstacle=False, joint_limits=None):
    robot = RobotModel.from_dh(
        [
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        ],
        name="planar2r",
    )
    world = SimpleWorld(robot)
    if with_obstacle:
        world.add_obstacle((0.0, 1.3), 0.1)
    return RobotWorkcell(robot, world, joint_limits=joint_limits)


def _targets(robot, configurations):
    return [robot.fk(q)[:3, 3] for q in configurations]


def test_workcell_solves_continuous_tcp_ik_and_plans_path():
    workcell = _workcell()
    q_start = np.array([0.2, -0.1])
    expected = np.array([[0.3, -0.2], [0.4, -0.3]])
    targets = _targets(workcell.robot, expected)

    ik = workcell.solve_tcp_path(targets, q_start, ik_kwargs={"max_iter": 300})
    task = workcell.plan_tcp_task(targets, q_start, bounds=[[-np.pi, np.pi], [-np.pi, np.pi]], ik_kwargs={"max_iter": 300})

    assert ik["joint_targets"].shape == (2, 2)
    assert np.max(ik["position_errors"]) < 1e-4
    assert np.allclose(task["joint_path"][0], q_start)
    assert np.allclose(task["joint_path"][-1], ik["joint_targets"][-1])
    assert not any(workcell.world.hit(q) for q in task["joint_path"])


def test_workcell_uses_rrt_when_direct_path_is_blocked():
    workcell = _workcell()
    workcell.world.add_obstacle((1.5, 0.0), 0.15)
    q_start = np.array([np.pi / 2.0, 0.0])
    q_goal = np.array([-np.pi / 2.0, 0.0])

    path = workcell.plan_joint_path(
        q_start,
        q_goal,
        bounds=[[-np.pi, np.pi], [-np.pi, np.pi]],
        step=0.15,
        goal_rate=0.25,
        max_iter=3000,
        random_state=0,
    )

    assert path.shape[0] > 2
    assert np.allclose(path[0], q_start)
    assert np.allclose(path[-1], q_goal)
    assert workcell._collision_free_path(path)


def test_workcell_enforces_joint_limits_and_reports_tracking_violations():
    limits = np.array([[-0.6, 0.6], [-0.5, 0.5]])
    workcell = _workcell(joint_limits=limits)
    path = workcell.plan_joint_path(np.array([-0.4, 0.0]), np.array([0.4, 0.2]))
    trajectory = workcell.retime_joint_path(path, velocity_limits=0.5)

    assert np.all(trajectory["position"] >= limits[:, 0])
    assert np.all(trajectory["position"] <= limits[:, 1])
    with pytest.raises(ValueError, match="joint_limits"):
        workcell.plan_joint_path(np.array([-0.4, 0.0]), np.array([0.7, 0.0]))
    with pytest.raises(ValueError, match="joint_limits"):
        workcell.retime_joint_path(np.array([[0.0, 0.0], [0.7, 0.0]]), velocity_limits=0.5)
    target_outside_limits = workcell.robot.fk(np.array([0.7, 0.0]))[:3, 3]
    with pytest.raises(ValueError, match="joint_limits"):
        workcell.solve_tcp_path([target_outside_limits], np.array([0.4, 0.0]), ik_kwargs={"max_iter": 300})

    report = workcell.safety_report(
        trajectory,
        tracking={
            "joint_error": np.zeros((2, 2)),
            "actual": np.array([[0.0, 0.0], [0.7, 0.0]]),
        },
    )

    assert report["joint_limit_violation"]
    assert report["maximum_joint_limit_violation"] == pytest.approx(0.1)
    assert report["joint_limits"] == limits.tolist()


def test_workcell_retimes_tracks_reports_and_exports(tmp_path):
    workcell = _workcell(with_obstacle=True)
    path = np.array([[0.2, -0.1], [0.35, -0.25], [0.5, -0.2]])
    velocity_limits = np.array([0.7, 0.6])
    acceleration_limits = np.array([1.2, 1.0])
    jerk_limits = np.array([5.0, 4.0])

    trajectory = workcell.retime_joint_path(
        path,
        velocity_limits,
        acceleration_limits,
        jerk_limits,
        num_samples_per_segment=101,
    )
    tracking = workcell.simulate_tracking(trajectory, gains=(5.0, 0.5))
    report = workcell.safety_report(trajectory, tracking)
    paths = workcell.export_artifacts(tmp_path, trajectory, tracking=tracking, report=report)

    assert np.allclose(trajectory["position"][0], path[0])
    assert np.allclose(trajectory["position"][-1], path[-1])
    assert np.all(np.max(np.abs(trajectory["velocity"]), axis=0) <= velocity_limits + 1e-12)
    assert np.all(np.max(np.abs(trajectory["acceleration"]), axis=0) <= acceleration_limits + 1e-12)
    assert np.all(np.max(np.abs(trajectory["jerk"]), axis=0) <= jerk_limits + 1e-12)
    assert tracking["actual"].shape == tracking["reference"].shape
    assert np.all(np.isfinite(tracking["joint_error"]))
    assert report["minimum_clearance"] is not None
    assert report["minimum_clearance"] > 0.0
    assert not report["collision"]
    assert report["tracking_rms_error"] >= 0.0

    with paths["trajectory_csv"].open() as handle:
        rows = list(csv.reader(handle))
    assert rows[0][0] == "time"
    assert len(rows) == trajectory["time"].size + 1
    exported_report = json.loads(paths["safety_report_json"].read_text())
    assert exported_report["workcell"] == "planar2r"
    assert paths["tracking_csv"].is_file()


def test_workcell_plans_continuous_cartesian_targets_with_clearance():
    workcell = _workcell(with_obstacle=True)
    q_start = np.array([0.2, -0.1])
    q_goal = np.array([0.45, -0.25])
    target = workcell.robot.fk(q_goal)[:3, 3]

    task = workcell.plan_cartesian_task(
        [target],
        q_start,
        steps_per_segment=6,
        ik_kwargs={"max_iter": 300},
        clearance=0.02,
    )

    assert len(task["interpolated_targets"]) == 6
    assert task["joint_path"].shape == (7, 2)
    assert np.all(np.asarray([workcell.world.clearance(q) for q in task["joint_path"]]) >= 0.02)
    assert np.all(np.asarray(task["ik"]["position_errors"]) < 1e-4)


def test_workcell_report_includes_motion_and_singularity_diagnostics():
    workcell = _workcell()
    path = np.array([[0.2, -0.1], [0.35, -0.25], [0.5, -0.2]])
    trajectory = workcell.retime_joint_path(path, velocity_limits=0.7, acceleration_limits=1.2, jerk_limits=5.0)
    report = workcell.safety_report(trajectory, clearance_margin=0.0)

    assert report["clearance_violation"] is False
    assert report["motion_limit_violation"] is False
    assert report["motion_limits"]["velocity"]["maximum_by_joint"]
    assert report["minimum_position_manipulability"] >= 0.0
    assert np.isfinite(report["maximum_position_condition_number"])
