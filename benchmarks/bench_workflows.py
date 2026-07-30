"""Benchmark the business workflow surfaces.

The goal is to keep a fast, deterministic smoke benchmark for the three
product-facing workflows: tabular readiness, signal health, and robot workcell
planning.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from mastermlx.base import BaseResult, export_reports
from mastermlx.data import DataContract
from mastermlx.robotics import RobotModel, RobotWorkcell
from mastermlx.signal import SignalHealthExperiment
from mastermlx.sim import SimpleWorld
from mastermlx.tabular import DataReadinessReport


def bench(fn, n_runs=3):
    times = []
    result = None
    for _ in range(n_runs):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return float(np.mean(times)), result


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def benchmark_tabular_readiness(verbose=True):
    train = np.array(
        [
            [20.0, 0.1],
            [32.0, 0.3],
            [45.0, 0.6],
            [58.0, 0.9],
        ]
    )
    incoming = np.array(
        [
            [24.0, 0.2],
            [121.0, 0.5],
            [39.0, np.nan],
            [39.0, np.nan],
        ]
    )
    contract = DataContract(
        rules={
            "x0": {"kind": "numeric", "min": 0.0, "max": 120.0},
            "x1": {"kind": "numeric", "min": 0.0, "max": 1.0, "missing_rate": 0.25},
        }
    )
    readiness = DataReadinessReport(data_contract=contract).fit(train)
    elapsed, report = bench(lambda: readiness.run(incoming), n_runs=5)
    if verbose:
        print(
            f"  readiness report        {elapsed:8.5f}s  "
            f"status={report.status}  issues={len(report.issues)}"
        )
    return elapsed, report


def benchmark_signal_health(verbose=True):
    sample_rate = 1000
    t = np.arange(2048, dtype=float) / sample_rate
    signal = np.sin(2.0 * np.pi * 50.0 * t) + 0.15 * np.sin(2.0 * np.pi * 180.0 * t)
    experiment = SignalHealthExperiment(
        sample_rate=sample_rate,
        feature_limits={"rms": (0.1, 1.2), "dominant_frequency": (40.0, 70.0)},
        bands=[(0.0, 100.0), (100.0, 250.0)],
        window_length=256,
        hop_length=128,
    )
    elapsed, report = bench(lambda: experiment.run(signal), n_runs=5)
    n_windows = 0 if report.windows is None else report.windows["features"].shape[0]
    if verbose:
        print(
            f"  signal health           {elapsed:8.5f}s  "
            f"status={report.summary['status']}  score={report.summary['health_score']:.1f}  "
            f"windows={n_windows}"
        )
    return elapsed, report


def benchmark_robot_workcell(verbose=True):
    robot = RobotModel.from_dh(
        [
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        ],
        name="benchmark-planar2r",
    )
    world = SimpleWorld(robot)
    workcell = RobotWorkcell(robot, world, joint_limits=[[-np.pi, np.pi], [-np.pi, np.pi]])
    elapsed, result = bench(
        lambda: workcell.plan_motion(
            q_start=[0.2, -0.1],
            q_goal=[0.5, -0.2],
            planner="rrt_star",
            velocity_limits=0.8,
            acceleration_limits=1.5,
            jerk_limits=8.0,
            track=True,
            tracking_kwargs={"dt": 0.05},
            stop_on_first_path=True,
        ),
        n_runs=3,
    )
    if verbose:
        print(
            f"  robot workcell          {elapsed:8.5f}s  "
            f"waypoints={result.planning_report['n_waypoints']}  "
            f"duration={result.trajectory.duration:.3f}s"
        )
    return elapsed, result


def run_workflow_suite(output_dir=None, *, verbose=True):
    """Run all business workflow smoke checks and optionally export reports."""

    if verbose:
        section("Tabular Readiness")
    tabular_time, tabular = benchmark_tabular_readiness(verbose=verbose)

    if verbose:
        section("Signal Health")
    signal_time, signal = benchmark_signal_health(verbose=verbose)

    if verbose:
        section("Robot Workcell")
    robot_time, robot = benchmark_robot_workcell(verbose=verbose)

    reports = BaseResult(
        {
            "tabular_readiness": tabular,
            "signal_health": signal,
            "robot_workcell": robot.safety_report,
        }
    )
    timings = BaseResult(
        {
            "tabular_readiness": tabular_time,
            "signal_health": signal_time,
            "robot_workcell": robot_time,
        }
    )
    artifacts = None if output_dir is None else export_reports(reports, output_dir)
    return BaseResult({"reports": reports, "timings": timings, "artifacts": artifacts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="optional directory for JSON report artifacts")
    args = parser.parse_args()

    result = run_workflow_suite(output_dir=args.output)
    section("Summary")
    print("  Business workflow smoke benchmarks completed.")
    if result.artifacts is not None:
        print(f"  reports exported to {result.artifacts.manifest.parent}")


if __name__ == "__main__":
    main()
