"""Benchmark inspection route planning and coverage evaluation."""

from __future__ import annotations

import argparse
import time

import numpy as np

from mastermlx.robotics import RobotModel, RobotWorkcell
from mastermlx.sim import SimpleWorld


def benchmark_inspection(n_scan_poses=8, n_points=128, runs=3, verbose=True):
    robot = RobotModel.from_dh(
        [
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        ],
        name="inspection-benchmark-2r",
    )
    workcell = RobotWorkcell(robot, SimpleWorld(robot))
    configurations = np.column_stack([
        np.linspace(-0.4, 0.4, n_scan_poses),
        np.linspace(-0.35, -0.15, n_scan_poses),
    ])
    scan_poses = robot.fk_batch(configurations)[:, :3, 3]
    rng = np.random.default_rng(0)
    points = scan_poses[rng.integers(0, n_scan_poses, size=n_points)]
    points = points + rng.normal(0.0, 0.025, size=points.shape)
    durations = []
    result = None
    for _ in range(int(runs)):
        start = time.perf_counter()
        result = workcell.plan_inspection(
            scan_poses,
            [-0.5, 0.0],
            inspection_points=points,
            coverage_radius=0.1,
            velocity_limits=0.8,
        )
        durations.append(time.perf_counter() - start)
    report = result["inspection_report"]
    summary = {
        "scan_poses": int(n_scan_poses),
        "inspection_points": int(n_points),
        "seconds": float(np.mean(durations)),
        "reachability_rate": float(report["reachability_rate"]),
        "coverage_rate": float(report["coverage_rate"]),
        "occlusion_rate": float(report["occlusion_rate"]),
        "scan_duration": report["scan_duration"],
    }
    if verbose:
        print(f"scan poses           {n_scan_poses:8d}")
        print(f"inspection points    {n_points:8d}")
        print(f"planning seconds     {summary['seconds']:8.5f}")
        print(f"reachability         {summary['reachability_rate']:.3f}")
        print(f"coverage             {summary['coverage_rate']:.3f}")
        print(f"occlusion            {summary['occlusion_rate']:.3f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-poses", type=int, default=8)
    parser.add_argument("--points", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    benchmark_inspection(args.scan_poses, args.points, args.runs)


if __name__ == "__main__":
    main()
