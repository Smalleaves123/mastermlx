import numpy as np
import pytest

from mastermlx.robotics import RobotExperiment, RobotModel, RobotWorkcell
from mastermlx.sim import SimpleWorld


def _robot():
    return RobotModel.from_dh(
        [
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        ],
        name="api-planar2r",
    )


def test_robot_model_exposes_canonical_aliases_and_batch_methods():
    robot = _robot()
    configurations = np.array([[0.0, 0.0], [0.2, -0.1]])

    assert robot.n_joints == 2
    assert np.allclose(robot.forward_kinematics(configurations[0]), robot.fk(configurations[0]))
    assert np.allclose(robot.inverse_kinematics([2.0, 0.0, 0.0], joint_values=[0.1, -0.1], max_iter=200), robot.ik([2.0, 0.0, 0.0], joint_values=[0.1, -0.1], max_iter=200))
    assert robot.fk_batch(configurations).shape == (2, 4, 4)
    assert robot.positions_batch(configurations).shape == (2, 3)
    assert robot.jacobian_batch(configurations).shape == (2, 6, 2)
    assert np.allclose(
        robot.jacobian_batch(configurations),
        np.asarray([robot.jacobian(q) for q in configurations]),
    )
    with pytest.raises(ValueError, match="finite"):
        robot.fk([0.0, np.nan])


def test_robot_model_joint_limits_and_constrained_ik():
    limits = np.array([[-0.8, 0.8], [-0.8, 0.8]])
    robot = _robot()
    robot = RobotModel.from_dh(robot.links, name="limited", joint_limits=limits)
    assert np.allclose(robot.joint_limits, limits)
    assert np.allclose(robot.default_joint_values(), [0.0, 0.0])
    assert np.allclose(robot.clip_joint_values([2.0, -2.0]), [0.8, -0.8])
    assert np.allclose(robot.joint_limit_violation([2.0, -2.0]), 1.2)

    target = robot.fk([0.45, -0.35])[:3, 3]
    solution = robot.ik(target, joint_values=[0.0, 0.0], max_iter=300, damping=1e-5)
    assert robot.joint_limit_violation(solution) == 0.0
    assert np.allclose(robot.fk(solution)[:3, 3], target, atol=1e-5)
    with pytest.raises(ValueError, match="joint_limits"):
        robot.fk([1.0, 0.0])

    offset_robot = RobotModel.from_dh(robot.links, joint_limits=[[1.0, 2.0], [1.0, 2.0]])
    assert np.allclose(offset_robot.default_joint_values(), [1.5, 1.5])
    assert np.allclose(offset_robot.fk(), offset_robot.fk([1.5, 1.5]))


def test_robot_model_velocity_mapping_and_differential_ik():
    robot = _robot()
    q = np.array([0.2, -0.1])
    qd = np.array([0.3, -0.2])

    desired = robot.end_effector_velocity(q, qd, translational=True)
    recovered = robot.differential_ik(
        q,
        desired,
        damping=1e-8,
        translational=True,
    )

    assert desired.shape == (3,)
    assert recovered.shape == q.shape
    assert np.allclose(
        robot.end_effector_velocity(q, recovered, translational=True),
        desired,
        atol=1e-7,
    )
    with pytest.raises(ValueError, match="task_velocity"):
        robot.differential_ik(q, [1.0, 2.0], translational=True)
    with pytest.raises(ValueError, match="damping"):
        robot.differential_ik(q, desired, damping=-1.0, translational=True)


def test_robot_model_batch_velocity_mapping():
    robot = _robot()
    configurations = np.array([[0.2, -0.1], [0.1, 0.3]])
    velocities = np.array([[0.3, -0.2], [-0.1, 0.4]])

    batch = robot.end_effector_velocity_batch(configurations, velocities, translational=True)
    expected = np.asarray(
        [robot.end_effector_velocity(q, qd, translational=True) for q, qd in zip(configurations, velocities)]
    )
    assert batch.shape == (2, 3)
    assert np.allclose(batch, expected)
    with pytest.raises(ValueError, match="same shape"):
        robot.end_effector_velocity_batch(configurations, velocities[:1])


def test_robot_results_support_mapping_and_attribute_access(tmp_path):
    robot = _robot()
    world = SimpleWorld(robot)
    experiment = RobotExperiment(robot)
    trajectory = experiment.plan_trajectory([0.0, 0.0], [0.2, 0.1], duration=1.0, num_waypoints=3, num_samples_per_segment=4)
    workcell = RobotWorkcell(robot, world)
    result = workcell.retime_joint_path(trajectory[1], velocity_limits=1.0)
    report = workcell.safety_report(result)

    assert result["position"] is result.position
    assert result.n_samples == result["position"].shape[0]
    assert result.n_joints == 2
    assert result.duration > 0.0
    assert report["workcell"] == report.workcell
    assert workcell.export_artifacts(tmp_path, result)["trajectory_csv"].is_file()
