from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
        track=False,
    )

    print("waypoints:", result.planning_report["n_waypoints"])
    print("path length:", round(result.planning_report["joint_path_length"], 3))
    print("reference minimum clearance:", round(result.safety_report["reference_minimum_clearance"], 3))
    print("execution_ready:", result.safety_report["execution_ready"])

    trajectory = result.trajectory
    reference_positions = robot.fk_batch(trajectory["position"])[:, :3, 3]
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "robotics"
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    world.render([np.pi / 2.0, 0.0], ax=axes[0], annotate=True)
    axes[0].plot(reference_positions[:, 0], reference_positions[:, 1], label="planned TCP")
    axes[0].set_title("Collision-free workcell path")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].plot(trajectory["time"], trajectory["position"])
    axes[1].set_title("Retimed joint trajectory")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("joint position (rad)")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    output_path = output_dir / "workcell_planning.png"
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    print("plot:", output_path)


if __name__ == "__main__":
    main()
