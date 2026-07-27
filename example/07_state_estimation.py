"""Planar pose estimation with odometry and sensor updates."""

import numpy as np

from common import check_release
from mastermlx.robotics import PlanarPoseEKF, wrap_angle


check_release()
filter_ = PlanarPoseEKF(
    x0=np.array([0.0, 0.0, 0.0]),
    P0=np.eye(3) * 0.1,
    Q=np.eye(3) * 0.01,
    R_heading=np.array([[0.02]]),
    R_position=np.eye(2) * 0.05,
)

for _ in range(5):
    filter_.predict(odometry=[0.2, 0.05], dt=0.1)

filter_.update_heading(0.03)
filter_.update_position([0.1, 0.0])
state = filter_.step(
    odometry=[0.1, 0.0],
    dt=0.1,
    pose=[0.12, 0.01, 0.03],
)

print("state:", state)
print("position:", filter_.position)
print("yaw:", filter_.yaw)
print("covariance:\n", filter_.covariance)
print("wrapped_angle:", wrap_angle(3.5))
