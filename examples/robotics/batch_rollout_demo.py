"""Compare independent Python rollouts with the batched NumPy path."""

from __future__ import annotations

import time

import numpy as np

from mastermlx.robotics import RobotModel
from mastermlx.sim import SimpleRobotSim


def main():
    robot = RobotModel.from_dh(
        [
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        ]
    )
    rng = np.random.default_rng(0)
    actions = rng.normal(0.0, 0.2, size=(64, 128, robot.n_joints))
    sim = SimpleRobotSim(robot, dt=0.02, damping=0.1)

    start = time.perf_counter()
    single = []
    for sequence in actions:
        sim.reset()
        single.append(sim.rollout(sequence)[0])
    single_seconds = time.perf_counter() - start

    start = time.perf_counter()
    batched = sim.rollout_batch(actions)
    batch_seconds = time.perf_counter() - start

    single = np.asarray(single)
    print("single_shape:", single.shape)
    print("batch_shape:", batched.shape)
    print("single_seconds:", round(single_seconds, 5))
    print("batch_seconds:", round(batch_seconds, 5))
    print("speedup:", round(single_seconds / batch_seconds, 2))
    print("parity_error:", np.max(np.abs(single - batched)))


if __name__ == "__main__":
    main()
