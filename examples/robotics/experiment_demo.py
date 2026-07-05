from __future__ import annotations

import numpy as np

from mastermlx.robotics import PlanarPoseEKF, RobotExperiment


def planar_2r_urdf():
    return """
    <robot name="planar2r">
      <link name="base" />
      <link name="link1" />
      <link name="link2" />
      <joint name="joint1" type="revolute">
        <parent link="base" />
        <child link="link1" />
        <origin xyz="1 0 0" rpy="0 0 0" />
        <axis xyz="0 0 1" />
      </joint>
      <joint name="joint2" type="revolute">
        <parent link="link1" />
        <child link="link2" />
        <origin xyz="1 0 0" rpy="0 0 0" />
        <axis xyz="0 0 1" />
      </joint>
    </robot>
    """


def main():
    estimator = PlanarPoseEKF(
        x0=np.array([0.0, 0.0, 0.0]),
        P0=np.eye(3),
        Q=1e-4 * np.eye(3),
        R_heading=np.array([[0.05]]),
        R_position=0.05 * np.eye(2),
    )
    experiment = RobotExperiment.from_urdf(planar_2r_urdf(), name="planar2r", pose_estimator=estimator)

    print("fk:", experiment.fk([0.0, 0.0])[:3, 3])
    print("jacobian shape:", experiment.jacobian([0.2, -0.1]).shape)
    print("ik:", experiment.ik(np.array([1.7, 0.3, 0.0]), joint_values=[0.1, -0.1], max_iter=200))

    times, positions, velocities, accelerations = experiment.plan_trajectory(
        np.array([0.0, 0.0]),
        np.array([0.7, -0.4]),
        duration=2.0,
        num_waypoints=5,
        num_samples_per_segment=6,
        smoothness=0.5,
    )
    report = experiment.trajectory_report(positions)
    print("trajectory samples:", times.shape[0])
    print("trajectory report:", report["joint_path_length"], report["end_effector_path_length"])

    states, poses, controls = experiment.track_trajectory(positions, gains=(2.0, 0.1), dt=0.05)
    print("tracking states:", states.shape)
    print("tracking controls:", controls.shape)
    print("final pose:", poses[-1][:3, 3])

    predicted, _ = experiment.estimate_pose(np.array([1.0, 0.0]), dt=1.0)
    print("pose estimator:", predicted)
    print("summary:", experiment.summary())


if __name__ == "__main__":
    main()
