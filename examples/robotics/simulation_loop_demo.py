"""Run a controller inside the standard robot simulation environment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mastermlx.robotics import RobotModel
from mastermlx.sim import RobotSimulation, SimpleWorld


def main():
    robot = RobotModel.from_dh(
        [
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        ],
        name="simulation-planar2r",
    )
    world = SimpleWorld(robot)
    world.add_obstacle(center=(1.0, 0.75), radius=0.12)
    goal = np.array([0.45, -0.55])
    target = robot.fk(goal)[:3, 3]
    env = RobotSimulation(
        world,
        dt=0.02,
        damping=0.15,
        max_steps=400,
        target_position=target,
        target_tolerance=0.03,
        terminate_on_collision=False,
    )

    observation, reset_info = env.reset(seed=0)
    tcp_trace = [observation["tcp_position"].copy()]
    distance_trace = []
    for _ in range(400):
        q_error = goal - observation["q"]
        action = 8.0 * q_error - 1.0 * observation["qd"]
        observation, _, terminated, truncated, info = env.step(
            {"joint": action, "gripper": 1.0}
        )
        tcp_trace.append(observation["tcp_position"].copy())
        distance_trace.append(info["distance_to_target"])
        if terminated or truncated:
            break

    trace = np.asarray(tcp_trace)
    print("seed:", reset_info["seed"])
    print("steps:", info["step"])
    print("termination:", info.get("termination_reason", "running"))
    print("success:", info["success"])
    print("final_distance:", round(info["distance_to_target"], 4))

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "robotics"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    world.render(observation["q"], ax=axes[0], annotate=True)
    axes[0].plot(trace[:, 0], trace[:, 1], label="TCP trace")
    axes[0].scatter(target[0], target[1], marker="*", s=120, label="target")
    axes[0].set_title("Closed-loop simulation")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend()
    axes[1].plot(np.arange(len(distance_trace)) * env.dt, distance_trace)
    axes[1].set_title("TCP distance to target")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("distance")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    output_path = output_dir / "simulation_loop.png"
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    print("plot:", output_path)


if __name__ == "__main__":
    main()
