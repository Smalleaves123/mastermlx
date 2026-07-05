import numpy as np

from mastermlx.robotics import PlanarPoseEKF, RobotExperiment, compare_robot_models


def _planar_2r_dh():
    return [
        {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
    ]


def test_robot_experiment_supports_kinematics_and_planning():
    experiment = RobotExperiment.from_dh(_planar_2r_dh(), name="planar2r")

    pose = experiment.fk([0.0, 0.0])
    q = experiment.ik(np.array([2.0, 0.0, 0.0]), joint_values=[0.1, -0.1], max_iter=200)
    times, positions, velocities, accelerations = experiment.plan_trajectory(
        np.array([0.0, 0.0]),
        np.array([1.0, 0.5]),
        duration=1.5,
        num_waypoints=4,
        num_samples_per_segment=4,
    )
    report = experiment.trajectory_report(positions)

    assert np.allclose(pose[:3, 3], np.array([2.0, 0.0, 0.0]))
    assert np.allclose(experiment.fk(q)[:3, 3], np.array([2.0, 0.0, 0.0]), atol=1e-4)
    assert times.ndim == 1
    assert positions.shape[1] == 2
    assert velocities.shape == positions.shape
    assert accelerations.shape == positions.shape
    assert report["n_joints"] == 2
    assert report["end_effector_path_length"] >= 0.0
    assert experiment.summary()["name"] == "planar2r"


def test_robot_experiment_tracks_trajectory_and_updates_pose_estimator():
    experiment = RobotExperiment.from_dh(
        _planar_2r_dh(),
        name="planar2r-ekf",
        pose_estimator=PlanarPoseEKF(
            x0=np.array([0.0, 0.0, 0.0]),
            P0=np.eye(3),
            Q=1e-4 * np.eye(3),
            R_heading=np.array([[0.05]]),
            R_position=0.05 * np.eye(2),
        ),
    )
    trajectory = np.array([
        [0.0, 0.0],
        [0.2, 0.1],
        [0.4, 0.2],
    ])
    states, poses, controls = experiment.track_trajectory(trajectory, gains=(2.0, 0.1), dt=0.05)
    predicted, _ = experiment.estimate_pose(np.array([1.0, 0.0]), dt=1.0)

    assert states.shape == (4, 4)
    assert len(poses) == 4
    assert controls.shape == (3, 2)
    assert np.all(np.isfinite(states))
    assert predicted.shape == (3,)


def test_compare_robot_models_returns_leaderboard():
    shorter = RobotExperiment.from_dh(
        [
            {"a": 0.5, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 0.5, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        ],
        name="short",
    ).robot
    longer = RobotExperiment.from_dh(
        [
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        ],
        name="long",
    ).robot

    result = compare_robot_models([("short", shorter), ("long", longer)], joint_values=[0.0, 0.0])

    assert result["leaderboard"]
    assert result["best_name"] in {"short", "long"}
