import numpy as np

from mastermlx.robotics import optimize_joint_path


def test_joint_path_optimizer_returns_motion_limit_feasible_trajectory():
    result = optimize_joint_path(
        np.array([[0.0, 0.0], [0.8, -0.4], [1.0, 0.2]]),
        smoothness=0.5,
        velocity_limits=[0.4, 0.5],
        acceleration_limits=[0.8, 1.0],
        jerk_limits=[4.0, 5.0],
        num_samples_per_segment=51,
    )
    trajectory = result["trajectory"]

    assert trajectory["velocity_limits"].shape == (2,)
    assert np.max(np.abs(trajectory["velocity"]), axis=0)[0] <= 0.4 + 1e-10
    assert np.max(np.abs(trajectory["velocity"]), axis=0)[1] <= 0.5 + 1e-10
    assert np.max(np.abs(trajectory["acceleration"]), axis=0)[0] <= 0.8 + 1e-10
    assert np.max(np.abs(trajectory["jerk"]), axis=0)[1] <= 5.0 + 1e-10
    assert np.allclose(trajectory["position"][0], [0.0, 0.0])
    assert np.allclose(trajectory["position"][-1], [1.0, 0.2])
