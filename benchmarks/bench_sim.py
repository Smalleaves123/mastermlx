"""Benchmark single and batched lightweight robot simulation rollouts."""

from __future__ import annotations

import argparse
import time

import numpy as np

from mastermlx.robotics import RobotModel
from mastermlx.sim import RobotSimulation, SimpleRobotSim, SimpleWorld


def _robot():
    return RobotModel.from_dh(
        [
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        ],
        name="simulation-benchmark-2r",
    )


def _timed(fn, runs):
    durations = []
    result = None
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        durations.append(time.perf_counter() - start)
    return float(np.mean(durations)), result


def benchmark_simulation(batch_size=128, horizon=256, runs=3, verbose=True):
    rng = np.random.default_rng(0)
    actions = rng.normal(0.0, 0.2, size=(batch_size, horizon, 2))
    robot = _robot()

    def single_rollout():
        sim = SimpleRobotSim(robot, dt=0.02, damping=0.1)
        results = []
        for sequence in actions:
            sim.reset()
            results.append(sim.rollout(sequence)[0])
        return results

    def batch_rollout():
        sim = SimpleRobotSim(robot, dt=0.02, damping=0.1)
        return sim.rollout_batch(actions)

    single_time, single_states = _timed(single_rollout, runs)
    batch_time, batch_states = _timed(batch_rollout, runs)
    speedup = single_time / batch_time if batch_time > 0.0 else np.inf
    result = {
        "batch_size": int(batch_size),
        "horizon": int(horizon),
        "single_seconds": single_time,
        "batch_seconds": batch_time,
        "speedup": float(speedup),
        "parity_error": float(
            max(
                np.max(np.abs(batch_states[index] - single_states[index]))
                for index in range(batch_size)
            )
        ),
    }
    if verbose:
        print(f"single rollout       {single_time:8.5f}s")
        print(f"batched rollout      {batch_time:8.5f}s")
        print(f"batch speedup        {speedup:8.2f}x")
        print(f"parity error         {result['parity_error']:.3e}")
    return result


def benchmark_environment(horizon=128, runs=3, verbose=True):
    robot = _robot()
    world = SimpleWorld(robot)
    env = RobotSimulation(world, dt=0.02, damping=0.1, max_steps=horizon + 1)
    actions = np.zeros((horizon, robot.n_joints), dtype=float)
    elapsed, rollout = _timed(lambda: env.rollout(actions), runs)
    result = {"seconds": elapsed, "steps": int(rollout["rewards"].size)}
    if verbose:
        print(f"environment rollout  {elapsed:8.5f}s  steps={result['steps']}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=256)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    print("Lightweight robot simulation benchmark")
    benchmark_simulation(args.batch_size, args.horizon, args.runs)
    benchmark_environment(min(args.horizon, 128), args.runs)


if __name__ == "__main__":
    main()
