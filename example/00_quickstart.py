"""The shortest useful mastermlx robotics example."""

import numpy as np

from common import check_release
from mastermlx.robotics import DHLink, RobotModel


check_release()
robot = RobotModel.from_dh(
    [
        DHLink(a=1.0, alpha=0.0, d=0.0, theta=0.0),
        DHLink(a=1.0, alpha=0.0, d=0.0, theta=0.0),
    ],
    name="two-link-arm",
)
q = np.array([0.2, -0.1])
pose = robot.forward_kinematics(q)
print("end_effector_position:", pose[:3, 3])
print("pose_shape:", pose.shape)
