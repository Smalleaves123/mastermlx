"""Profile collision scaling, hit-buffer capacity, and batched dynamics."""

from __future__ import annotations

import time
import tracemalloc

import numpy as np

from mastermlx import get_backend, set_backend
from mastermlx.robotics import (
    LinkInertia,
    RobotModel,
    BoxObstacle,
    SphereObstacle,
    chain_clearance_batch,
    chain_collision_details_batch,
    chain_collision_free_batch,
)


def _measure(fn, runs=3):
    fn()
    times = []
    peak = 0
    for _ in range(runs):
        tracemalloc.start()
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
        _, current_peak = tracemalloc.get_traced_memory()
        peak = max(peak, current_peak)
        tracemalloc.stop()
    return float(np.mean(times)), int(peak)


def _robot():
    links = [
        {"a": 0.24 + 0.02 * (index % 3), "alpha": 0.0, "d": 0.0, "theta": 0.0}
        for index in range(7)
    ]
    inertias = [
        LinkInertia(
            mass=1.0 + 0.1 * index,
            center_of_mass=(-0.5 * link["a"], 0.0, 0.0),
            inertia=(0.01, 0.01, 0.02),
        )
        for index, link in enumerate(links)
    ]
    return RobotModel.from_dh(links, name="scaling-robot", link_inertias=inertias)


def _details_output(samples, capacity):
    return {
        "minimum_clearance": np.empty(samples),
        "collision": np.empty(samples, dtype=bool),
        "closest_kind": np.empty(samples, dtype=np.int8),
        "closest_index": np.empty(samples, dtype=np.int64),
        "closest_obstacle_index": np.empty(samples, dtype=np.int64),
        "hit_count": np.empty(samples, dtype=np.int64),
        "hit_truncated": np.empty(samples, dtype=bool),
        "hit_kind": np.empty((samples, capacity), dtype=np.int8),
        "hit_index": np.empty((samples, capacity), dtype=np.int64),
        "hit_obstacle_index": np.empty((samples, capacity), dtype=np.int64),
        "hit_clearance": np.empty((samples, capacity)),
    }


def main():
    robot = _robot()
    rng = np.random.default_rng(0)
    configurations = rng.normal(0.0, 0.45, size=(2_000, robot.n_joints))
    points = robot.frame_positions_batch(configurations)[:, :, :2]
    near = [SphereObstacle((0.6, 0.0), 0.08)]
    old = get_backend()
    try:
        set_backend("auto")
        print("AABB broad-phase scaling (2000 chains, distant obstacles):")
        for count in (0, 16, 64, 256):
            obstacles = near + [
                BoxObstacle((50.0 + index, 50.0), (50.1 + index, 50.1))
                for index in range(count)
            ]
            exact, _ = _measure(lambda: chain_clearance_batch(points, obstacles, link_radius=0.02))
            broadphase, _ = _measure(
                lambda: chain_collision_free_batch(points, obstacles, clearance=0.0, link_radius=0.02)
            )
            print(
                f"  obstacles={count + len(near):3d}  exact={exact:.5f}s  "
                f"broadphase={broadphase:.5f}s  speedup={exact / broadphase:.2f}x"
            )

        obstacles = near + [SphereObstacle((50.0 + index, 50.0), 0.05) for index in range(63)]
        default_capacity = len(obstacles) * (2 * points.shape[1] - 1)
        bounded_capacity = 8
        default_time, default_peak = _measure(
            lambda: chain_collision_details_batch(points, obstacles, link_radius=0.02)
        )
        bounded_time, bounded_peak = _measure(
            lambda: chain_collision_details_batch(
                points, obstacles, link_radius=0.02, max_hits=bounded_capacity
            )
        )
        output = _details_output(points.shape[0], bounded_capacity)
        reused_time, reused_peak = _measure(
            lambda: chain_collision_details_batch(
                points,
                obstacles,
                link_radius=0.02,
                max_hits=bounded_capacity,
                output=output,
            )
        )
        print("Detailed hit-buffer capacity (64 obstacles):")
        print(f"  default slots={default_capacity:4d}  time={default_time:.5f}s  peak={default_peak / 1e6:.2f} MB")
        print(f"  bounded slots={bounded_capacity:4d}  time={bounded_time:.5f}s  peak={bounded_peak / 1e6:.2f} MB")
        print(f"  reused  slots={bounded_capacity:4d}  time={reused_time:.5f}s  peak={reused_peak / 1e6:.2f} MB")

        velocities = np.zeros_like(configurations)
        accelerations = np.full_like(configurations, 0.2)
        set_backend("numpy")
        numpy_mass_time, _ = _measure(lambda: robot.mass_matrix_batch(configurations))
        numpy_torque_time, _ = _measure(
            lambda: robot.inverse_dynamics_batch(configurations, velocities, accelerations)
        )
        set_backend("auto")
        auto_mass_time, _ = _measure(lambda: robot.mass_matrix_batch(configurations))
        auto_torque_time, _ = _measure(
            lambda: robot.inverse_dynamics_batch(configurations, velocities, accelerations)
        )
        set_backend("numpy")
        forward_torques = robot.inverse_dynamics_batch(
            configurations, velocities, accelerations
        )
        numpy_forward_time, _ = _measure(
            lambda: robot.forward_dynamics_batch(configurations, velocities, forward_torques)
        )
        set_backend("auto")
        auto_forward_time, _ = _measure(
            lambda: robot.forward_dynamics_batch(configurations, velocities, forward_torques)
        )
        coriolis_configurations = configurations[:256]
        coriolis_velocities = velocities[:256]
        set_backend("numpy")
        numpy_coriolis_time, _ = _measure(
            lambda: robot.coriolis_forces_batch(coriolis_configurations, coriolis_velocities), runs=1
        )
        set_backend("auto")
        auto_coriolis_time, _ = _measure(
            lambda: robot.coriolis_forces_batch(coriolis_configurations, coriolis_velocities), runs=1
        )
        print("Batched dynamics (2000 x 7-DOF configurations):")
        print(
            f"  mass_matrix  numpy={numpy_mass_time:.5f}s  auto={auto_mass_time:.5f}s  "
            f"speedup={numpy_mass_time / auto_mass_time:.2f}x"
        )
        print(
            f"  inverse_dynamics  numpy={numpy_torque_time:.5f}s  auto={auto_torque_time:.5f}s  "
            f"speedup={numpy_torque_time / auto_torque_time:.2f}x"
        )
        print(
            f"  forward_dynamics  numpy={numpy_forward_time:.5f}s  auto={auto_forward_time:.5f}s  "
            f"speedup={numpy_forward_time / auto_forward_time:.2f}x"
        )
        print(
            f"  coriolis (256 x 7-DOF)  numpy={numpy_coriolis_time:.5f}s  "
            f"auto={auto_coriolis_time:.5f}s  speedup={numpy_coriolis_time / auto_coriolis_time:.2f}x"
        )
    finally:
        set_backend(old)


if __name__ == "__main__":
    main()
