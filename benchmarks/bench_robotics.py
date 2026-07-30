"""Benchmark batched robot kinematics and velocity-level helpers."""

from __future__ import annotations

import time

import numpy as np

from mastermlx import get_backend, set_backend
from mastermlx.robotics import (
    BoxObstacle,
    CapsuleObstacle,
    RobotModel,
    SphereObstacle,
    chain_clearance_batch,
    chain_collision_details_batch,
    chain_collision_free_batch,
    chain_collision_summary_batch,
    compose_transform_batch,
    homogeneous_transform,
    interpolate_pose_batch,
    robotics_backend_report,
    trajectory_peaks_batch,
)
from mastermlx.robotics import RobotWorkcell
from mastermlx.sim import SimpleWorld


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
    chain_points = np.asarray([robot.positions(values)[:, :2] for values in configurations])
    obstacles = [
        SphereObstacle((0.5, 0.1), 0.08),
        BoxObstacle((0.9, -0.2), (1.1, 0.2)),
        CapsuleObstacle((0.0, 0.6), (1.8, 0.6), 0.05),
    ]
    broadphase_obstacles = obstacles + [SphereObstacle((20.0 + index, 20.0), 0.1) for index in range(20)]
    workcell = RobotWorkcell(robot, SimpleWorld(robot))
    planning_robot = RobotModel.from_dh(
        [
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        ],
        name="planning-benchmark-robot",
    )
    planning_world = SimpleWorld(planning_robot)
    planning_world.add_obstacle((1.5, 0.0), 0.15)
    planning_workcell = RobotWorkcell(planning_robot, planning_world)
    retime_path = configurations[:32]
    ik_targets = robot.positions_batch(configurations[:64])
    ik_seeds = np.zeros((ik_targets.shape[0], robot.n_joints), dtype=float)
    transform_pairs = np.asarray(
        [
            [
                homogeneous_transform(np.eye(3), [0.1, 0.0, 0.0]),
                homogeneous_transform(np.eye(3), [0.0, 0.2, 0.0]),
            ]
        ]
        * configurations.shape[0]
    )
    pose_start = homogeneous_transform(np.eye(3), [0.0, 0.0, 0.0])
    pose_end = homogeneous_transform(
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        [1.0, 0.5, 0.2],
    )
    pose_alphas = np.linspace(0.0, 1.0, configurations.shape[0])
    trajectory_metrics = rng.normal(0.0, 0.5, size=(configurations.shape[0], robot.n_joints, 3))
    print(f"Robot backend requested: {get_backend()}")
    print(f"Robot backend capabilities: {robotics_backend_report()}")
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
            clearance_time, clearances = bench(
                lambda: chain_clearance_batch(chain_points, obstacles, link_radius=0.02)
            )
            summary_time, summary = bench(
                lambda: chain_collision_summary_batch(chain_points, obstacles, link_radius=0.02)
            )
            details_time, details = bench(
                lambda: chain_collision_details_batch(chain_points, obstacles, link_radius=0.02)
            )
            broadphase_time, free = bench(
                lambda: chain_collision_free_batch(
                    chain_points,
                    broadphase_obstacles,
                    clearance=0.0,
                    link_radius=0.02,
                )
            )
            compose_time, composed = bench(lambda: compose_transform_batch(transform_pairs))
            interpolate_time, interpolated_poses = bench(
                lambda: interpolate_pose_batch(pose_start, pose_end, pose_alphas)
            )
            metrics_time, metrics = bench(
                lambda: robot.kinematic_metrics_batch(configurations, translational=True)
            )
            peaks_time, peaks = bench(lambda: trajectory_peaks_batch(trajectory_metrics))
            planning_time, planning_path = bench(
                lambda: planning_workcell.plan_joint_path(
                    [np.pi / 2.0, 0.0],
                    [-np.pi / 2.0, 0.0],
                    bounds=[[-np.pi, np.pi], [-np.pi, np.pi]],
                    planner="rrt_star",
                    step=0.15,
                    goal_rate=0.25,
                    max_iter=600,
                    random_state=0,
                    stop_on_first_path=True,
                    smooth_path=False,
                    workers=2,
                )
            )
            retime_time, retimed = bench(
                lambda: workcell.retime_joint_path(
                    retime_path,
                    velocity_limits=0.8,
                    acceleration_limits=1.5,
                    jerk_limits=8.0,
                    num_samples_per_segment=21,
                )
            )
            ik_time, ik_solutions = bench(
                lambda: robot.ik_batch(
                    ik_targets,
                    joint_values=ik_seeds,
                    warm_start=False,
                    max_iter=50,
                    damping=1e-3,
                )
            )
            print(
                f"{backend:>5}  fk_batch={fk_time:.5f}s ({poses.shape})  "
                f"positions_batch={positions_time:.5f}s ({positions.shape})  "
                f"jacobian_batch={jacobian_time:.5f}s ({jacobians.shape})  "
                f"velocity_batch={velocity_batch_time:.5f}s ({velocity_batch.shape})  "
                f"velocity={velocity_time:.5f}s ({velocity.shape})  "
                f"clearance_batch={clearance_time:.5f}s ({clearances.shape})  "
                f"collision_summary={summary_time:.5f}s ({summary['minimum_clearance'].shape})  "
                f"collision_details={details_time:.5f}s ({details['hit_kind'].shape})  "
                f"broadphase={broadphase_time:.5f}s ({free.shape})  "
                f"compose_batch={compose_time:.5f}s ({composed.shape})  "
                f"pose_interp={interpolate_time:.5f}s ({interpolated_poses.shape})  "
                f"kinematic_metrics_batch={metrics_time:.5f}s ({metrics['condition_number'].shape})  "
                f"trajectory_peaks={peaks_time:.5f}s ({peaks.shape})  "
                f"planning_workers={planning_time:.5f}s ({planning_path.shape})  "
                f"retime={retime_time:.5f}s ({retimed['position'].shape})  "
                f"ik_batch={ik_time:.5f}s ({ik_solutions.shape})"
            )
    finally:
        set_backend(old)


if __name__ == "__main__":
    main()
