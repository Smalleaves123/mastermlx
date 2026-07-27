"""Print the public robotics API grouped by the task it supports."""

import inspect

from common import check_release
import mastermlx.robotics as robotics


check_release()
groups = {
    "model_and_kinematics": [
        "DHLink", "RobotModel", "forward_kinematics", "geometric_jacobian",
        "inverse_kinematics", "chain_positions", "finite_difference_jacobian",
    ],
    "transforms": [
        "compose_transform", "homogeneous_transform", "invert_transform",
        "transform_points", "matrix_to_quaternion", "quaternion_to_matrix",
        "matrix_to_euler", "euler_to_matrix", "rpy_to_matrix", "rot_x",
        "rot_y", "rot_z", "skew", "unskew",
    ],
    "trajectory": [
        "cubic_time_scaling", "quintic_time_scaling", "joint_trajectory",
        "sample_joint_trajectory", "sample_joint_trajectory_segments",
        "plan_joint_path", "smooth_joint_path", "plan_joint_trajectory",
    ],
    "workflows": [
        "RobotExperiment", "RobotWorkcell", "RobotResult", "JointTrajectory",
        "parse_urdf", "urdf_to_dh_chain", "compare_robot_models",
    ],
    "estimation_and_visualization": [
        "PlanarPoseEKF", "wrap_angle", "plot_chain",
    ],
}

for group, names in groups.items():
    print(f"\n[{group}]")
    for name in names:
        obj = getattr(robotics, name)
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):
            signature = "class"
        print(f"{name}{signature}")

print("\nUse the grouped files in this directory for runnable examples of these APIs.")
