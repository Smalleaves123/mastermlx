"""Select camera viewpoints and plan a coverage-aware TCP inspection route."""

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
        name="active-inspection-planar2r",
    )
    world = SimpleWorld(robot)
    workcell = RobotWorkcell(robot, world)
    candidate_configurations = np.array(
        [
            [-0.45, -0.15],
            [-0.25, -0.30],
            [-0.05, -0.45],
            [0.15, -0.40],
            [0.35, -0.25],
            [0.50, -0.10],
        ]
    )
    candidates = [pose[:3, 3] for pose in robot.fk_batch(candidate_configurations)]
    candidate_points = np.asarray(candidates)
    inspection_points = np.asarray(
        [
            0.5 * (candidate_points[0] + candidate_points[1]),
            0.5 * (candidate_points[2] + candidate_points[3]),
            0.5 * (candidate_points[4] + candidate_points[5]),
        ]
    )
    result = workcell.plan_active_inspection(
        inspection_points,
        q_start=candidate_configurations[0],
        candidate_poses=candidates,
        coverage_radius=0.55,
        dwell_time=0.05,
        travel_speed=0.5,
        velocity_limits=0.8,
        acceleration_limits=1.5,
        jerk_limits=8.0,
    )
    report = result["active_inspection_report"]
    selected = result["selected_candidate_indices"]
    visibility = result["candidate_visibility"]["visible"]
    print("selected candidate indices:", selected.tolist())
    print("coverage_rate:", round(report["coverage_rate"], 3))
    print("candidate_occlusion_rate:", round(report["candidate_occlusion_rate"], 3))
    print("reachability_rate:", round(report["candidate_reachability_rate"], 3))
    print("total_inspection_time:", round(report["total_inspection_time"], 3))

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "robotics"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    world.render(candidate_configurations[-1], ax=axes[0], annotate=True)
    axes[0].scatter(candidate_points[:, 0], candidate_points[:, 1], c="0.65", s=55, label="candidate TCP")
    if selected.size:
        route = candidate_points[selected]
        axes[0].plot(route[:, 0], route[:, 1], "-o", c="tab:blue", label="selected scan route")
    axes[0].scatter(
        inspection_points[:, 0],
        inspection_points[:, 1],
        c=report["covered_points"],
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        s=90,
        marker="s",
        label="surface samples",
    )
    axes[0].set_title("Active inspection: candidates and selected TCP route")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(loc="best")

    image = axes[1].imshow(visibility, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title("Candidate camera visibility")
    axes[1].set_xlabel("inspection-point index")
    axes[1].set_ylabel("candidate-view index")
    axes[1].set_yticks(np.arange(visibility.shape[0]))
    axes[1].set_xticks(np.arange(visibility.shape[1]))
    for candidate_index, point_index in np.argwhere(visibility):
        axes[1].text(point_index, candidate_index, "✓", ha="center", va="center", color="black")
    figure.colorbar(image, ax=axes[1], ticks=[0, 1], label="visible")
    figure.suptitle(
        f"coverage={report['coverage_rate']:.0%}, "
        f"reachable={report['candidate_reachability_rate']:.0%}, "
        f"time={report['total_inspection_time']:.2f}s"
    )
    figure.tight_layout()
    output_path = output_dir / "active_inspection_planning.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("plot:", output_path)


if __name__ == "__main__":
    main()
