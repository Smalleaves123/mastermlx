import numpy as np

from mastermlx.robotics import PlanarPoseEKF


def test_planar_pose_ekf_predict_and_heading_update():
    ekf = PlanarPoseEKF(
        x0=np.array([0.0, 0.0, 0.0]),
        P0=np.eye(3),
        Q=1e-4 * np.eye(3),
        R_heading=np.array([[0.05]]),
        R_position=0.05 * np.eye(2),
        R_pose=np.eye(3),
    )

    predicted, _ = ekf.predict(np.array([1.0, 0.0]), dt=1.0)
    assert np.allclose(predicted[:2], np.array([1.0, 0.0]), atol=1e-6)

    updated, _ = ekf.update_heading(np.array(np.pi / 2))
    assert 0.0 < updated[2] < np.pi / 2


def test_planar_pose_ekf_position_update_accepts_partial_measurements():
    ekf = PlanarPoseEKF(
        x0=np.array([1.0, 1.0, 0.0]),
        P0=np.eye(3),
        Q=1e-4 * np.eye(3),
        R_heading=np.array([[0.1]]),
        R_position=0.01 * np.eye(2),
    )

    before = ekf.state.copy()
    updated, _ = ekf.update_position(np.array([2.0, 2.0]))

    assert np.linalg.norm(updated[:2] - np.array([2.0, 2.0])) < np.linalg.norm(before[:2] - np.array([2.0, 2.0]))
