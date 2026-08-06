"""Simulate grasp, TCP-relative transport, and release of one object."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mastermlx.robotics import RobotModel
from mastermlx.sim import RobotSimulation, SimpleWorld


def _drive(env, observation, goal, gripper, steps, tcp_trace, object_trace):
    for _ in range(steps):
        action = 7.0 * (goal - observation["q"]) - 0.8 * observation["qd"]
        observation, _, terminated, truncated, info = env.step(
            {"joint": action, "gripper": gripper}
        )
        tcp_trace.append(observation["tcp_position"].copy())
        object_trace.append(observation["object_positions"][0].copy())
        if terminated or truncated:
            break
    return observation, info


def main():
    robot = RobotModel.from_dh(
        [
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        ],
        name="pick-place-planar2r",
    )
    world = SimpleWorld(robot)
    part = world.add_object("part", robot.fk([0.0, 0.0])[:3, 3], radius=0.06)
    env = RobotSimulation(world, dt=0.02, damping=0.15, max_steps=300, grasp_distance=0.15)
    observation, _ = env.reset(seed=1)
    tcp_trace = [observation["tcp_position"].copy()]
    object_trace = [observation["object_positions"][0].copy()]

    observation, info = _drive(
        env, observation, np.array([0.0, 0.0]), gripper=0.0, steps=5,
        tcp_trace=tcp_trace, object_trace=object_trace,
    )
    print("grasp_event:", info.get("event"))
    print("attached_after_grasp:", part.attached)
    observation, info = _drive(
        env, observation, np.array([0.55, -0.75]), gripper=0.0, steps=100,
        tcp_trace=tcp_trace, object_trace=object_trace,
    )
    observation, _, _, _, info = env.step({"joint": [0.0, 0.0], "gripper": 1.0})
    tcp_trace.append(observation["tcp_position"].copy())
    object_trace.append(observation["object_positions"][0].copy())
    print("release_event:", info.get("event"))
    print("attached_after_release:", part.attached)
    print("released_position:", np.round(part.position, 4))

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "robotics"
    output_dir.mkdir(parents=True, exist_ok=True)
    tcp_trace = np.asarray(tcp_trace)
    object_trace = np.asarray(object_trace)
    figure, axis = plt.subplots(figsize=(7, 6))
    world.render(observation["q"], ax=axis, annotate=True)
    axis.plot(tcp_trace[:, 0], tcp_trace[:, 1], label="TCP")
    axis.plot(object_trace[:, 0], object_trace[:, 1], "--", label="part")
    axis.scatter(object_trace[0, 0], object_trace[0, 1], marker="o", s=80, label="pick")
    axis.scatter(object_trace[-1, 0], object_trace[-1, 1], marker="*", s=120, label="release")
    axis.set_title("Pick-and-place simulation")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal", adjustable="box")
    axis.legend()
    figure.tight_layout()
    output_path = output_dir / "pick_place_simulation.png"
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    print("plot:", output_path)


if __name__ == "__main__":
    main()
