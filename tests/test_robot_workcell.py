import csv
import json

import numpy as np
import pytest

from mastermlx.control import JointPDController
from mastermlx.robotics import LinkInertia, RobotModel, RobotWorkcell, URDFRobotModel
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


def test_workcell_path_collision_summary_uses_batched_chain_queries():
    workcell = _workcell()
    workcell.world.add_obstacle((1.5, 0.0), 0.15)
    path = np.array([[np.pi / 2.0, 0.0], [-np.pi / 2.0, 0.0]])

    summary = workcell.path_collision_summary(path, interpolation_step=0.05)

    assert summary["collision"]
    assert summary["first_collision_index"] is not None
    assert summary["n_samples"] > path.shape[0]
    assert not workcell.path_collision_free(path, interpolation_step=0.05)


def test_workcell_parallel_planning_is_seed_deterministic():
    workcell = _workcell()
    workcell.world.add_obstacle((1.5, 0.0), 0.15)
    kwargs = dict(
        bounds=[[-np.pi, np.pi], [-np.pi, np.pi]],
        step=0.15,
        goal_rate=0.25,
        max_iter=3000,
        random_state=0,
        planner="rrt_star",
        stop_on_first_path=True,
        smooth_path=False,
    )
    serial = workcell.plan_joint_path(
        [np.pi / 2.0, 0.0], [-np.pi / 2.0, 0.0], workers=1, **kwargs
    )
    parallel = workcell.plan_joint_path(
        [np.pi / 2.0, 0.0], [-np.pi / 2.0, 0.0], workers=3, **kwargs
    )
    assert np.array_equal(parallel, serial)


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
    assert tracking["controller_status"]["steps"] == tracking["reference"].shape[0]
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


def test_workcell_plan_motion_returns_planning_trajectory_and_reports():
    workcell = _workcell()

    result = workcell.plan_motion(
        np.array([0.2, -0.1]),
        np.array([0.5, -0.2]),
        bounds=[[-np.pi, np.pi], [-np.pi, np.pi]],
        planner="rrt_star",
        step=0.2,
        goal_rate=0.25,
        search_radius=0.5,
        max_iter=1200,
        random_state=3,
        stop_on_first_path=True,
        velocity_limits=0.8,
        acceleration_limits=1.5,
        jerk_limits=8.0,
        track=True,
        tracking_kwargs={"dt": 0.05},
    )

    assert np.allclose(result["path"][0], [0.2, -0.1])
    assert np.allclose(result["path"][-1], [0.5, -0.2])
    assert result["planning_report"]["planner"] == "rrt_star"
    assert result["planning_report"]["n_waypoints"] == result["path"].shape[0]
    assert not result["planning_report"]["collision"]
    assert result["trajectory"].n_joints == 2
    assert result["tracking"]["actual"].shape == result["tracking"]["reference"].shape
    assert not result["safety_report"]["reference_collision"]
    assert workcell.report_ is result["safety_report"]
    assert workcell.artifacts_["motion"] is result


def test_workcell_plans_pick_and_place_with_time_aligned_gripper_events():
    workcell = _workcell()
    q_start = np.array([0.2, -0.1])
    pick_target = workcell.robot.fk([0.4, -0.3])[:3, 3]
    place_target = workcell.robot.fk([0.55, -0.45])[:3, 3]

    task = workcell.plan_pick_and_place(
        pick_target,
        place_target,
        q_start,
        bounds=[[-np.pi, np.pi], [-np.pi, np.pi]],
        approach_offset=[-0.05, 0.0, 0.0],
        steps_per_segment=4,
        ik_kwargs={"max_iter": 300},
        velocity_limits=0.7,
    )

    assert np.allclose(task["joint_path"][0], q_start)
    assert np.allclose(task["trajectory"]["path"], task["joint_path"])
    assert task["phase_indices"]["pick_grasp"] < task["phase_indices"]["place_release"]
    assert [event["command"] for event in task["gripper_schedule"]] == ["open", "close", "open"]
    assert np.all(np.diff([event["time"] for event in task["gripper_schedule"]]) >= 0.0)
    assert not task["safety_report"]["collision"]
    assert workcell.artifacts_["pick_and_place"] is task


def test_workcell_accepts_spatial_urdf_and_runs_motion_workflow():
    xml = """
    <robot name="spatial_workcell">
      <link name="base" />
      <link name="tip" />
      <joint name="slide" type="prismatic">
        <parent link="base" /><child link="tip" />
        <origin xyz="0 0 0" rpy="0 0 0" />
        <axis xyz="1 0 0" />
        <limit lower="0" upper="1" />
      </joint>
    </robot>
    """
    robot = URDFRobotModel.from_urdf(xml, name="spatial_workcell")
    workcell = RobotWorkcell(robot)
    result = workcell.plan_motion(
        [0.1],
        [0.8],
        bounds=[[0.0, 1.0]],
        velocity_limits=0.8,
        acceleration_limits=1.0,
        jerk_limits=4.0,
    )

    assert result["trajectory"].n_joints == 1
    assert result["safety_report"]["valid"]
    assert result["safety_report"]["execution_ready"]
    assert result["safety_report"]["workcell"] == "spatial_workcell"


def test_workcell_spatial_urdf_supports_tcp_ik_alias():
    xml = """
    <robot name="spatial_tcp">
      <link name="base" />
      <link name="tip" />
      <joint name="slide" type="prismatic">
        <parent link="base" /><child link="tip" />
        <origin xyz="0 0 0" rpy="0 0 0" />
        <axis xyz="1 0 0" />
        <limit lower="0" upper="1" />
      </joint>
    </robot>
    """
    robot = URDFRobotModel.from_urdf(xml, name="spatial_tcp")
    workcell = RobotWorkcell(robot)
    target = robot.fk([0.6])[:3, 3]

    result = workcell.solve_tcp_path([target], [0.1], ik_kwargs={"max_iter": 100})

    assert result["joint_targets"].shape == (1, 1)
    assert np.allclose(result["joint_targets"], [[0.6]], atol=1e-5)


def test_validate_trajectory_is_an_execution_gate():
    workcell = _workcell()
    workcell.world.add_obstacle((0.0, 0.0), 0.2)
    unsafe = {
        "time": np.array([0.0, 0.0]),
        "position": np.array([[0.0, 0.0], [0.1, 0.0]]),
        "velocity": np.zeros((2, 2)),
        "acceleration": np.zeros((2, 2)),
        "jerk": np.zeros((2, 2)),
    }

    report = workcell.validate_trajectory(unsafe)

    assert not report["valid"]
    assert not report["execution_ready"]
    assert "invalid_time" in report["violations"]
    assert "collision" in report["violations"]
    with pytest.raises(RuntimeError, match="collision"):
        workcell.validate_trajectory(unsafe, raise_on_failure=True)


def test_self_collision_is_reported_and_can_be_excluded():
    robot = RobotModel.from_dh(
        [
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        ],
        name="self_collision_robot",
    )
    workcell = RobotWorkcell(robot)
    path = np.array([[0.0, np.pi, np.pi]])

    summary = workcell.path_collision_summary(path, check_self_collision=True)
    excluded = RobotWorkcell(robot, self_collision_exclusions=((0, 2),))
    excluded_summary = excluded.path_collision_summary(path, check_self_collision=True)

    assert summary["self_collision"]
    assert summary["collision"]
    assert not excluded_summary["self_collision"]
    report = workcell.validate_trajectory(
        {"time": [0.0], "position": path},
        check_self_collision=True,
    )
    assert not report["valid"]
    assert "self_collision" in report["violations"]


def test_workcell_accepts_external_controller_objects():
    workcell = _workcell()
    controller = JointPDController(2, kp=2.0, kd=0.1, output_limits=(-1.0, 1.0))

    tracking = workcell.simulate_tracking(
        np.zeros((4, 2), dtype=float),
        controller=controller,
        dt=0.05,
    )

    assert tracking["controller"] == "pd"
    assert tracking["controller_status"]["steps"] == 4
    assert tracking["controls"].shape == (4, 2)


def test_workcell_simulates_computed_torque_as_physical_torque():
    robot = RobotModel.from_dh(
        [{"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0}],
        link_inertias=[LinkInertia(mass=1.0, center_of_mass=(-0.5, 0.0, 0.0), inertia=(0.0, 0.0, 0.1))],
    )
    workcell = RobotWorkcell(robot, SimpleWorld(robot))

    tracking = workcell.simulate_tracking(
        np.array([[0.2]], dtype=float),
        controller="computed_torque",
        gains=(2.0, 0.1),
        dt=0.05,
    )

    assert tracking["controller"] == "computed_torque"
    assert tracking["controller_status"]["command_type"] == "torque"
    assert tracking["controls"].shape == (1, 1)
    assert np.all(np.isfinite(tracking["actual"]))
