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
    assert robot.jacobian_batch(configurations).shape == (2, 6, 2)
    with pytest.raises(ValueError, match="finite"):
        robot.fk([0.0, np.nan])


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
