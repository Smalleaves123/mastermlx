import numpy as np
import pytest

from mastermlx import get_backend, set_backend
from mastermlx.robotics import RobotModel, RobotWorkcell, robotics_backend_report
from mastermlx.robotics.trajectory import _load_cpp_retiming
from mastermlx.sim import SimpleWorld


def _workcell():
    robot = RobotModel.from_dh(
        [
            {"a": 0.8, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 0.6, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 0.4, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        ],
        name="retiming-cpp",
    )
    return RobotWorkcell(robot, SimpleWorld(robot))


def test_robotics_backend_report_includes_retiming_kernel():
    report = robotics_backend_report()
    assert "cpp_retiming" in report
    assert isinstance(report["cpp_retiming"], bool)


def test_cpp_retiming_matches_numpy_for_all_limits():
    cpp = _load_cpp_retiming("auto")
    if cpp is None:
        pytest.skip("C++ retiming extension is unavailable")

    workcell = _workcell()
    path = np.array(
        [[0.0, 0.1, -0.2], [0.3, -0.2, 0.1], [0.5, 0.2, 0.4], [0.2, 0.4, 0.0]],
        dtype=float,
    )
    kwargs = {
        "velocity_limits": np.array([0.8, 0.7, 0.9]),
        "acceleration_limits": np.array([1.5, 1.2, 1.4]),
        "jerk_limits": np.array([8.0, 7.0, 9.0]),
        "num_samples_per_segment": 17,
        "minimum_duration": 2e-3,
    }
    old = get_backend()
    try:
        set_backend("numpy")
        reference = workcell.retime_joint_path(path, **kwargs)
        set_backend("auto")
        accelerated = workcell.retime_joint_path(path, **kwargs)
    finally:
        set_backend(old)

    for key in ("time", "position", "velocity", "acceleration", "jerk", "durations"):
        assert np.allclose(accelerated[key], reference[key], atol=1e-12)


def test_cpp_retiming_supports_velocity_only_limits():
    cpp = _load_cpp_retiming("auto")
    if cpp is None:
        pytest.skip("C++ retiming extension is unavailable")
    workcell = _workcell()
    trajectory = workcell.retime_joint_path(
        np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.3]]),
        velocity_limits=0.5,
        num_samples_per_segment=5,
    )
    assert trajectory["position"].shape == (5, 3)
    assert trajectory["acceleration_limits"] is None
    assert trajectory["jerk_limits"] is None
