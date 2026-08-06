"""Plan and plot a small TCP inspection route with business metrics."""

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
        name="inspection-planar2r",
    )
    world = SimpleWorld(robot)
    workcell = RobotWorkcell(robot, world)
    scan_configurations = np.array([
        [-0.35, -0.2],
        [-0.15, -0.35],
        [0.05, -0.45],
        [0.25, -0.4],
        [0.45, -0.25],
    ])
    scan_poses = robot.fk_batch(scan_configurations)[:, :3, 3]
    inspection_points = scan_poses + np.array([0.0, 0.04, 0.0])
    result = workcell.plan_inspection(
        scan_poses,
        q_start=[-0.5, 0.0],
        inspection_points=inspection_points,
        coverage_radius=0.08,
        dwell_time=0.05,
        required_coverage=1.0,
        velocity_limits=0.8,
        acceleration_limits=1.5,
        jerk_limits=8.0,
    )
    report = result["inspection_report"]
    print("execution_ready:", report["execution_ready"])
    print("reachability_rate:", round(report["reachability_rate"], 3))
    print("coverage_rate:", round(report["coverage_rate"], 3))
    print("occlusion_rate:", round(report["occlusion_rate"], 3))
    print("total_inspection_time:", round(report["total_inspection_time"], 3))
    print("violations:", report["violations"])

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "robotics"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    world.render(scan_configurations[-1], ax=axes[0], annotate=True)
    axes[0].plot(scan_poses[:, 0], scan_poses[:, 1], "--", label="scan TCP")
    axes[0].scatter(
        inspection_points[:, 0], inspection_points[:, 1],
        c=report["covered_points"], cmap="RdYlGn", vmin=0, vmax=1, s=70, label="inspection points",
    )
    axes[0].set_title("Inspection scan and coverage")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend()
    if result["trajectory"] is not None:
        axes[1].plot(result["trajectory"]["time"], result["trajectory"]["position"])
    axes[1].set_title("Retimed scan motion")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("joint position (rad)")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    output_path = output_dir / "inspection_simulation.png"
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    print("plot:", output_path)


if __name__ == "__main__":
    main()
