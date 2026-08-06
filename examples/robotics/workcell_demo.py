"""A small end-to-end Cartesian task using the released Workcell API."""

import numpy as np

from mastermlx.robotics import RobotModel, RobotWorkcell
from mastermlx.sim import SimpleWorld


robot = RobotModel.from_dh(
    [
        {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
    ],
    name="workcell-planar2r",
)
world = SimpleWorld(robot)
workcell = RobotWorkcell(robot, world)
q_start = np.array([0.0, 0.0])
q_goal = np.array([0.2, -0.1])
target = robot.fk(q_goal)[:3, 3]

tcp = workcell.solve_tcp_path(
    [target],
    q_start,
    ik_kwargs={"max_iter": 200},
    check_collisions=False,
)
print("tcp_result_keys:", list(tcp))
print("tcp_joint_targets:\n", tcp["joint_targets"])

cartesian = workcell.plan_cartesian_task(
    [target],
    q_start,
    steps_per_segment=4,
    ik_kwargs={"max_iter": 200},
    check_collisions=False,
)
print("cartesian_result_keys:", list(cartesian))

print("joint_path_shape:", cartesian["joint_path"].shape)
print("final_position:", robot.fk(cartesian["joint_path"][-1])[:3, 3])

trajectory = workcell.retime_joint_path(cartesian["joint_path"], velocity_limits=0.5)
report = workcell.safety_report(trajectory)
print("retimed_trajectory_keys:", list(trajectory))
print("safety_report_keys:", list(report))
print("trajectory_samples:", trajectory.n_samples)
print("trajectory_duration:", trajectory.duration)
print("minimum_clearance:", report.minimum_clearance)
