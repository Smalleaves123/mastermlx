import numpy as np
import pytest

from mastermlx import get_backend, set_backend
from mastermlx.robotics import RobotModel, robotics_backend_report
from mastermlx.robotics.kinematics import _load_cpp_ik


def _robot():
    return RobotModel.from_dh(
        [
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 0.8, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        ],
        name="ik-cpp",
    )


def test_robotics_backend_report_includes_batch_ik():
    report = robotics_backend_report()
    assert "cpp_ik" in report
    assert isinstance(report["cpp_ik"], bool)


def test_cpp_batch_ik_matches_numpy_with_warm_start_and_limits():
    cpp = _load_cpp_ik("auto")
    if cpp is None:
        pytest.skip("C++ IK extension is unavailable")

    robot = RobotModel.from_dh(
        _robot().links,
        joint_limits=[[-1.2, 1.2], [-1.2, 1.2]],
    )
    configurations = np.array([[0.15, -0.1], [0.2, -0.05], [0.25, 0.0], [0.3, 0.05]])
    targets = robot.positions_batch(configurations)
    kwargs = {"joint_values": [0.0, 0.0], "max_iter": 250, "damping": 1e-5}
    old = get_backend()
    try:
        set_backend("numpy")
        reference = robot.ik_batch(targets, **kwargs)
        set_backend("auto")
        accelerated = robot.ik_batch(targets, **kwargs)
    finally:
        set_backend(old)

    assert np.allclose(robot.positions_batch(accelerated), targets, atol=1e-5)
    assert np.allclose(accelerated, reference, atol=2e-5)


def test_cpp_batch_ik_supports_per_target_seeds_and_base_tool():
    cpp = _load_cpp_ik("auto")
    if cpp is None:
        pytest.skip("C++ IK extension is unavailable")

    base = np.array(
        [[0.0, -1.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.2], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    tool = np.array(
        [[1.0, 0.0, 0.0, 0.05], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    robot = RobotModel.from_dh(_robot().links, base=base, tool=tool)
    configurations = np.array([[0.1, -0.2], [0.25, -0.1], [0.35, 0.0]])
    targets = robot.positions_batch(configurations)
    seeds = np.zeros_like(configurations)
    old = get_backend()
    try:
        set_backend("numpy")
        reference = robot.ik_batch(targets, joint_values=seeds, warm_start=False, max_iter=250, damping=1e-5)
        set_backend("auto")
        solutions = robot.ik_batch(targets, joint_values=seeds, warm_start=False, max_iter=250, damping=1e-5)
    finally:
        set_backend(old)

    assert solutions.shape == configurations.shape
    assert np.allclose(solutions, reference, atol=2e-5)
