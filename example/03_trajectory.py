"""Joint-space time scaling, paths, smoothing, and sampled trajectories."""

import numpy as np

from common import check_release
from mastermlx.robotics import (
    cubic_time_scaling,
    joint_trajectory,
    plan_joint_path,
    plan_joint_trajectory,
    quintic_time_scaling,
    sample_joint_trajectory,
    sample_joint_trajectory_segments,
    smooth_joint_path,
)


check_release()
q0 = np.array([0.0, 0.5])
qf = np.array([1.0, 1.5])
s_cubic, ds_cubic, dds_cubic = cubic_time_scaling(2.0, 1.0)
s_quintic, ds_quintic, dds_quintic = quintic_time_scaling(2.0, 1.0)
print("cubic_scaling:", s_cubic, ds_cubic, dds_cubic)
print("quintic_scaling:", s_quintic, ds_quintic, dds_quintic)

position, velocity, acceleration = joint_trajectory(q0, qf, 2.0, 1.0)
print("single_sample:", position, velocity, acceleration)
times, positions, velocities, accelerations = sample_joint_trajectory(q0, qf, 2.0, num_samples=9)
print("sampled_shapes:", times.shape, positions.shape, velocities.shape, accelerations.shape)

waypoints = np.array([[0.0, 0.0], [0.7, 0.9], [1.0, 0.2]])
segment_times, segment_positions, _, _ = sample_joint_trajectory_segments(
    waypoints, [1.0, 1.5], num_samples_per_segment=5
)
print("segment_shapes:", segment_times.shape, segment_positions.shape)
path = plan_joint_path(q0, qf, num_waypoints=7, via_points=[[0.6, 1.0]])
smoothed = smooth_joint_path(path, smoothness=0.5)
print("path_shapes:", path.shape, smoothed.shape)

planned = plan_joint_trajectory(
    q0,
    qf,
    duration=2.0,
    num_waypoints=5,
    num_samples_per_segment=4,
)
print("planned_return_shapes:", [value.shape for value in planned])

print("trajectory_samples:", positions.shape[0])
print("planned_position_shape:", planned[1].shape)
