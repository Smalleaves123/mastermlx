"""Benchmark deterministic RRT and RRT* planning workloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from mastermlx.planning import rrt, rrt_star


BENCHMARK_SCHEMA = "mastermlx.planning-benchmark.v1"
DEFAULT_SEED = 7
DEFAULT_RUNS = 5


def _measure(function, runs):
    function()
    timings = []
    value = None
    for _ in range(runs):
        start = time.perf_counter()
        value = function()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings)), value


def run_planning_benchmark(*, seed=DEFAULT_SEED, runs=DEFAULT_RUNS):
    """Run fixed collision-free planning workloads and return a JSON-ready record."""

    seed = int(seed)
    runs = int(runs)
    if runs < 1:
        raise ValueError("runs must be at least 1")
    common = {
        "start": [0.05, 0.05],
        "goal": [0.95, 0.95],
        "bounds": [[0.0, 1.0], [0.0, 1.0]],
        "step": 0.04,
        "goal_rate": 0.1,
        "max_iter": 800,
        "random_state": seed,
    }

    rrt_seconds, rrt_path = _measure(lambda: rrt(**common), runs)
    rrt_star_seconds, rrt_star_path = _measure(
        lambda: rrt_star(
            **common,
            search_radius=0.12,
            goal_tolerance=0.04,
            stop_on_first_path=False,
        ),
        runs,
    )
    if rrt_path is None or rrt_star_path is None:
        raise RuntimeError("fixed planning workload did not find a path")

    return {
        "schema": BENCHMARK_SCHEMA,
        "seed": seed,
        "runs": runs,
        "workload": {
            "dimensions": 2,
            "max_iter": common["max_iter"],
            "step": common["step"],
            "search_radius": 0.12,
        },
        "results": {
            "rrt_seconds": rrt_seconds,
            "rrt_path_points": int(rrt_path.shape[0]),
            "rrt_star_seconds": rrt_star_seconds,
            "rrt_star_path_points": int(rrt_star_path.shape[0]),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    record = run_planning_benchmark(seed=args.seed, runs=args.runs)
    print(json.dumps(record, indent=2))
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
