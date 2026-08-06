"""DH kinematics, Jacobians, batch evaluation, and inverse kinematics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mastermlx.robotics import (
    DHLink,
    RobotModel,
    finite_difference_jacobian,
    forward_kinematics,
    geometric_jacobian,
    inverse_kinematics,
    planar_2r_jacobian,
    plot_chain,
)


links = [
    DHLink(a=1.0, alpha=0.0, d=0.0, theta=0.0),
    DHLink(a=1.0, alpha=0.0, d=0.0, theta=0.0),
]
q = np.array([0.25, -0.15])

pose = forward_kinematics(links, q)
jacobian = geometric_jacobian(links, q)
analytic_planar = planar_2r_jacobian(1.0, 1.0, q[0], q[1])
numeric = finite_difference_jacobian(
    lambda values: forward_kinematics(links, values)[:3, 3], q
)
print("single_state_pose_shape:", pose.shape)
print("jacobian_shape:", jacobian.shape)
print("analytic_planar_jacobian:\n", analytic_planar)
print("finite_difference_error:", np.max(np.abs(numeric - jacobian[:3])))

robot = RobotModel.from_dh(links, name="example-planar-2r")
configurations = np.array([[0.0, 0.0], [0.1, -0.1], [0.2, -0.05]])
poses = robot.fk_batch(configurations)
jacobians = robot.jacobian_batch(configurations)
print("batch_pose_shape:", poses.shape)
print("batch_jacobian_shape:", jacobians.shape)

target = pose[:3, 3]
solution = inverse_kinematics(target, links, joint_values=[0.1, -0.1], max_iter=200)
print("ik_solution:", solution)
print("ik_position_error:", np.linalg.norm(forward_kinematics(links, solution)[:3, 3] - target))

print("final_position:", pose[:3, 3])
print("jacobian_shape:", jacobian.shape)
chain_points = robot.positions(q)
plot = plot_chain(chain_points[:, :2], annotate=True)
print("plot_axes_type:", type(plot).__name__)
output_dir = Path(__file__).resolve().parents[1] / "outputs" / "robotics"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "kinematics_demo.png"
plot.get_figure().savefig(output_path, dpi=140, bbox_inches="tight")
plt.close(plot.get_figure())
print("plot:", output_path)
