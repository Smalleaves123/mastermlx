from __future__ import annotations

import numpy as np

from mastermlx.robotics import RobotModel, RobotWorkcell
from mastermlx.sim import SimpleWorld


def main():
    robot = RobotModel.from_dh(
        [
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        ],
        name="planar2r",
    )
    world = SimpleWorld(robot)
    world.add_obstacle((1.5, 0.0), 0.15)

    workcell = RobotWorkcell(
        robot,
        world,
        joint_limits=[[-np.pi, np.pi], [-np.pi, np.pi]],
    )
    result = workcell.plan_motion(
        q_start=[np.pi / 2.0, 0.0],
        q_goal=[-np.pi / 2.0, 0.0],
        planner="rrt",
        step=0.15,
        goal_rate=0.25,
        max_iter=3000,
        random_state=0,
        velocity_limits=0.8,
        acceleration_limits=1.5,
        jerk_limits=8.0,
        track=True,
        tracking_kwargs={"dt": 0.05},
    )

    print("waypoints:", result.planning_report["n_waypoints"])
    print("path length:", round(result.planning_report["joint_path_length"], 3))
    print("reference minimum clearance:", round(result.safety_report["reference_minimum_clearance"], 3))
    print("tracking RMS error:", round(result.safety_report["tracking_rms_error"], 4))


if __name__ == "__main__":
    main()
