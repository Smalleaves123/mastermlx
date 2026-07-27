"""Benchmark batched robot kinematics and velocity-level helpers."""

from __future__ import annotations

import time

import numpy as np

from mastermlx import get_backend, set_backend
from mastermlx.robotics import RobotModel


def bench(fn, n_runs=3):
    fn()
    timings = []
    result = None
    for _ in range(n_runs):
        start = time.perf_counter()
        result = fn()
        timings.append(time.perf_counter() - start)
    return float(np.mean(timings)), result


def make_robot(n_joints=7):
    links = [
        {
            "a": 0.25 + 0.02 * (index % 3),
            "alpha": 0.1 * ((index % 2) - 0.5),
            "d": 0.03 * (index % 2),
            "theta": 0.0,
        }
        for index in range(n_joints)
    ]
    return RobotModel.from_dh(links, name="benchmark-robot")


def main():
    robot = make_robot()
    rng = np.random.default_rng(0)
    configurations = rng.normal(0.0, 0.5, size=(2_000, robot.n_joints))
    q = configurations[0]
    qd = rng.normal(0.0, 0.2, size=robot.n_joints)
    print(f"Robot backend requested: {get_backend()}")
    old = get_backend()
    try:
        for backend in ("numpy", "auto"):
            set_backend(backend)
            fk_time, poses = bench(lambda: robot.fk_batch(configurations))
            positions_time, positions = bench(lambda: robot.positions_batch(configurations))
            jacobian_time, jacobians = bench(lambda: robot.jacobian_batch(configurations))
            velocity_batch_time, velocity_batch = bench(
                lambda: robot.end_effector_velocity_batch(configurations, np.tile(qd, (configurations.shape[0], 1)))
            )
            velocity_time, velocity = bench(
                lambda: robot.end_effector_velocity(q, qd, translational=False)
            )
            print(
                f"{backend:>5}  fk_batch={fk_time:.5f}s ({poses.shape})  "
                f"positions_batch={positions_time:.5f}s ({positions.shape})  "
                f"jacobian_batch={jacobian_time:.5f}s ({jacobians.shape})  "
                f"velocity_batch={velocity_batch_time:.5f}s ({velocity_batch.shape})  "
                f"velocity={velocity_time:.5f}s ({velocity.shape})"
            )
    finally:
        set_backend(old)


if __name__ == "__main__":
    main()
