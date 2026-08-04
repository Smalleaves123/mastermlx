"""Offline programming workflow for serial manipulators in a planar workcell."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..base import BaseExperiment
from ..planning import rrt_star, smooth
from .collision import (
    path_collision_free,
    path_collision_report,
    path_collision_summary,
    robot_collision_report,
)
from .constraints import validate_joint_limits
from .model import RobotModel
from .urdf_model import URDFRobotModel
from .results import JointTrajectory, RobotResult
from .trajectory import (
    _retime_quintic_path_compiled,
    optimize_joint_path as _optimize_joint_path,
    sample_joint_trajectory_segments,
    trajectory_peaks_batch,
)
from .transforms import homogeneous_transform, interpolate_pose_batch


_QUINTIC_MAX_VELOCITY = 1.875
_QUINTIC_MAX_ACCELERATION = 10.0 / np.sqrt(3.0)
_QUINTIC_MAX_JERK = 60.0


def _joint_vector(values, n_joints, name):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size != n_joints:
        raise ValueError(f"{name} must contain {n_joints} joint values")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _limits(values, n_joints, name):
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full(n_joints, float(values), dtype=float)
    else:
        values = values.reshape(-1)
    if values.size != n_joints:
        raise ValueError(f"{name} must be a scalar or contain {n_joints} values")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{name} must contain only positive finite values")
    return values


def _orientation_error(actual, target):
    rotation = target[:3, :3] @ actual[:3, :3].T
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _tcp_offset(target, offset, name):
    """Apply a world-frame TCP offset while preserving an optional orientation."""

    target = np.asarray(target, dtype=float)
    if target.shape == (3,):
        return target + offset
    if target.shape == (4, 4):
        result = target.copy()
        result[:3, 3] += offset
        return result
    raise ValueError(f"{name} must be a finite 3-vector or 4x4 transform")


def _task_offset(values, name):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite 3-vector")
    if np.linalg.norm(values) == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return values


class _SpatialWorkcellWorld:
    """Small obstacle world used when a spatial URDF has no custom world."""

    def __init__(self, robot, obstacles=(), occupancy_grid=None):
        self.robot = robot
        self.obstacles = list(obstacles)
        self.occupancy_grid = occupancy_grid

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)
        return obstacle

    def add_sphere(self, center, radius):
        from .collision import SphereObstacle

        return self.add_obstacle(SphereObstacle(tuple(np.asarray(center, dtype=float)), radius))

    def add_box(self, lower, upper):
        from .collision import BoxObstacle

        return self.add_obstacle(BoxObstacle(tuple(np.asarray(lower, dtype=float)), tuple(np.asarray(upper, dtype=float))))

    def add_capsule(self, start, end, radius):
        from .collision import CapsuleObstacle

        return self.add_obstacle(
            CapsuleObstacle(tuple(np.asarray(start, dtype=float)), tuple(np.asarray(end, dtype=float)), radius)
        )

    def link_positions(self, joint_values=None):
        return self.robot.frame_positions(joint_values)

    def collision_report(self, joint_values=None):
        return self.robot.collision_report(
            joint_values,
            self.obstacles,
            occupancy_grid=self.occupancy_grid,
        )

    def hit(self, joint_values=None):
        return bool(self.collision_report(joint_values)["collision"])

    def clearance(self, joint_values=None):
        return float(self.collision_report(joint_values)["minimum_clearance"])

    def plan_path(self, q_start, q_goal, bounds, *, hit=None, **kwargs):
        from ..planning import rrt

        return rrt(q_start, q_goal, bounds, hit=self.hit if hit is None else hit, **kwargs)

    def trajectory_follow(self, trajectory, *, gains=(4.0, 0.4), dt=0.1, damping=0.0, state=None):
        from ..sim.core import SimpleRobotSim

        sim = SimpleRobotSim(self.robot, state=state, dt=dt, damping=damping)
        states = [sim.state.copy()]
        poses = [sim.pose()]
        controls = []
        kp, kd = gains
        for target in np.asarray(trajectory, dtype=float):
            action = kp * (target - sim.q) + kd * (-sim.qd)
            controls.append(action)
            sim.step(action)
            states.append(sim.state.copy())
            poses.append(sim.pose())
        return np.asarray(states), poses, np.asarray(controls)


class RobotWorkcell(BaseExperiment):
    """Compose robot kinematics, collision planning, retiming, and tracking.

    DH robots retain the existing :class:`~mastermlx.sim.SimpleWorld` path;
    serial URDF robots use the same workflow with spatial obstacle queries.
    """

    def __init__(
        self,
        robot,
        world=None,
        name=None,
        joint_limits=None,
        occupancy_grid=None,
        self_collision_exclusions=(),
    ):
        from ..sim.world import SimpleWorld

        super().__init__()
        if not isinstance(robot, (RobotModel, URDFRobotModel)):
            raise TypeError("robot must be a RobotModel or URDFRobotModel")
        if world is None:
            world = (
                SimpleWorld(robot)
                if isinstance(robot, RobotModel)
                else _SpatialWorkcellWorld(robot, occupancy_grid=occupancy_grid)
            )
        if getattr(world, "robot", None) is not robot:
            raise ValueError("world.robot must be the same robot instance")
        if not hasattr(world, "obstacles"):
            raise TypeError("world must expose an obstacles collection")
        self.robot = robot
        self.world = world
        self.name = robot.name if name is None else str(name)
        self.occupancy_grid = (
            getattr(world, "occupancy_grid", None) if occupancy_grid is None else occupancy_grid
        )
        self.self_collision_exclusions = tuple(
            tuple(int(value) for value in pair) for pair in self_collision_exclusions
        )
        configured_limits = robot.joint_limits if joint_limits is None else joint_limits
        self.joint_limits = validate_joint_limits(configured_limits, self.n_joints)

    @property
    def n_joints(self):
        return self.robot.n_joints

    def _check_joint_limits(self, values, name):
        values = np.asarray(values, dtype=float)
        if values.shape[-1:] != (self.n_joints,):
            raise ValueError(f"{name} must end with {self.n_joints} joint values")
        if self.joint_limits is None:
            return
        if np.any(values < self.joint_limits[:, 0]) or np.any(values > self.joint_limits[:, 1]):
            raise ValueError(f"{name} exceeds configured joint_limits")

    def _resolve_bounds(self, bounds):
        if bounds is None:
            if self.joint_limits is None:
                raise ValueError("bounds are required when joint_limits are not configured")
            return self.joint_limits.copy()
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (self.n_joints, 2) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("bounds must have shape (n_joints, 2) with lower < upper")
        if self.joint_limits is None:
            return bounds
        clipped = np.column_stack(
            [
                np.maximum(bounds[:, 0], self.joint_limits[:, 0]),
                np.minimum(bounds[:, 1], self.joint_limits[:, 1]),
            ]
        )
        if np.any(clipped[:, 0] >= clipped[:, 1]):
            raise ValueError("bounds do not overlap configured joint_limits")
        return clipped

    def _joint_limit_violation(self, values):
        values = np.asarray(values, dtype=float)
        if self.joint_limits is None:
            return np.zeros(values.shape[:-1], dtype=float)
        lower = np.maximum(self.joint_limits[:, 0] - values, 0.0)
        upper = np.maximum(values - self.joint_limits[:, 1], 0.0)
        return np.max(np.maximum(lower, upper), axis=-1)

    def _state_collision_report(self, joint_values, *, link_radius=0.0, check_self_collision=False):
        if isinstance(self.robot, URDFRobotModel):
            return self.robot.collision_report(
                joint_values,
                self.world.obstacles,
                link_radius=link_radius,
                occupancy_grid=self.occupancy_grid,
                check_self_collision=check_self_collision,
                self_collision_exclusions=self.self_collision_exclusions,
            )
        return robot_collision_report(
            self.robot,
            joint_values,
            self.world.obstacles,
            link_radius=link_radius,
            occupancy_grid=self.occupancy_grid,
            check_self_collision=check_self_collision,
            self_collision_exclusions=self.self_collision_exclusions,
        )

    def _collision_free_path(
        self, path, collision_step=0.05, clearance=0.0, *, link_radius=0.0, check_self_collision=False
    ):
        if isinstance(self.robot, URDFRobotModel):
            return self.robot.path_collision_free(
                path,
                self.world.obstacles,
                clearance=clearance,
                link_radius=link_radius,
                interpolation_step=collision_step,
                occupancy_grid=self.occupancy_grid,
                check_self_collision=check_self_collision,
                self_collision_exclusions=self.self_collision_exclusions,
            )
        return path_collision_free(
            self.robot,
            path,
            self.world.obstacles,
            clearance=clearance,
            link_radius=link_radius,
            interpolation_step=collision_step,
            occupancy_grid=self.occupancy_grid,
            check_self_collision=check_self_collision,
            self_collision_exclusions=self.self_collision_exclusions,
        )

    def _collision_free_edge(
        self, start, end, collision_step, clearance, *, link_radius=0.0, check_self_collision=False
    ):
        return self._collision_free_path(
            np.vstack([start, end]),
            collision_step=collision_step,
            clearance=clearance,
            link_radius=link_radius,
            check_self_collision=check_self_collision,
        )

    def path_collision_summary(
        self,
        joint_path,
        *,
        link_radius=0.0,
        interpolation_step=0.05,
        check_self_collision=False,
    ):
        """Return compiled batch collision diagnostics for a joint-space path."""

        if isinstance(self.robot, URDFRobotModel):
            return self.robot.path_collision_summary(
                joint_path,
                self.world.obstacles,
                link_radius=link_radius,
                interpolation_step=interpolation_step,
                occupancy_grid=self.occupancy_grid,
                check_self_collision=check_self_collision,
                self_collision_exclusions=self.self_collision_exclusions,
            )
        return path_collision_summary(
            self.robot,
            joint_path,
            self.world.obstacles,
            link_radius=link_radius,
            interpolation_step=interpolation_step,
            occupancy_grid=self.occupancy_grid,
            check_self_collision=check_self_collision,
            self_collision_exclusions=self.self_collision_exclusions,
        )

    def path_collision_free(
        self,
        joint_path,
        *,
        clearance=0.0,
        link_radius=0.0,
        interpolation_step=0.05,
        check_self_collision=False,
    ):
        """Return whether a joint-space path satisfies an obstacle clearance."""

        if isinstance(self.robot, URDFRobotModel):
            return self.robot.path_collision_free(
                joint_path,
                self.world.obstacles,
                clearance=clearance,
                link_radius=link_radius,
                interpolation_step=interpolation_step,
                occupancy_grid=self.occupancy_grid,
                check_self_collision=check_self_collision,
                self_collision_exclusions=self.self_collision_exclusions,
            )
        return path_collision_free(
            self.robot,
            joint_path,
            self.world.obstacles,
            clearance=clearance,
            link_radius=link_radius,
            interpolation_step=interpolation_step,
            occupancy_grid=self.occupancy_grid,
            check_self_collision=check_self_collision,
            self_collision_exclusions=self.self_collision_exclusions,
        )

    def path_collision_report(
        self,
        joint_path,
        *,
        link_radius=0.0,
        interpolation_step=0.05,
        check_self_collision=False,
    ):
        """Return detailed collision diagnostics for a joint-space path."""

        if isinstance(self.robot, URDFRobotModel):
            return self.robot.path_collision_report(
                joint_path,
                self.world.obstacles,
                link_radius=link_radius,
                interpolation_step=interpolation_step,
                occupancy_grid=self.occupancy_grid,
                check_self_collision=check_self_collision,
                self_collision_exclusions=self.self_collision_exclusions,
            )
        return path_collision_report(
            self.robot,
            joint_path,
            self.world.obstacles,
            link_radius=link_radius,
            interpolation_step=interpolation_step,
            occupancy_grid=self.occupancy_grid,
            check_self_collision=check_self_collision,
            self_collision_exclusions=self.self_collision_exclusions,
        )

    def solve_tcp_path(
        self,
        targets,
        q_start,
        *,
        ik_kwargs=None,
        position_tolerance=1e-4,
        orientation_tolerance=1e-3,
        check_collisions=True,
        check_self_collision=False,
        link_radius=0.0,
    ):
        """Solve ordered TCP targets with each IK solution seeding the next one."""

        q_current = _joint_vector(q_start, self.n_joints, "q_start")
        self._check_joint_limits(q_current, "q_start")
        targets = list(targets)
        if not targets:
            raise ValueError("targets must be non-empty")
        ik_kwargs = {} if ik_kwargs is None else dict(ik_kwargs)
        position_tolerance = float(position_tolerance)
        orientation_tolerance = float(orientation_tolerance)
        if position_tolerance <= 0.0 or orientation_tolerance <= 0.0:
            raise ValueError("position_tolerance and orientation_tolerance must be positive")

        configurations = []
        position_errors = []
        orientation_errors = []
        normalized_targets = []
        for index, target in enumerate(targets):
            target = np.asarray(target, dtype=float)
            if target.shape not in {(3,), (4, 4)} or not np.all(np.isfinite(target)):
                raise ValueError("each target must be a finite 3-vector or 4x4 transform")
            q_current = _joint_vector(
                self.robot.ik(target, joint_values=q_current, **ik_kwargs),
                self.n_joints,
                f"IK solution for target {index}",
            )
            self._check_joint_limits(q_current, f"IK solution for target {index}")
            actual = self.robot.fk(q_current)
            position_error = float(np.linalg.norm(actual[:3, 3] - target[:3] if target.shape == (3,) else actual[:3, 3] - target[:3, 3]))
            orientation_error = 0.0 if target.shape == (3,) else _orientation_error(actual, target)
            if position_error > position_tolerance or orientation_error > orientation_tolerance:
                raise RuntimeError(
                    f"IK did not converge for TCP target {index}: "
                    f"position_error={position_error:.3e}, orientation_error={orientation_error:.3e}"
                )
            if check_collisions and self._state_collision_report(
                q_current,
                link_radius=link_radius,
                check_self_collision=check_self_collision,
            )["collision"]:
                raise RuntimeError(f"IK solution for TCP target {index} is in collision")
            normalized_targets.append(target.copy())
            configurations.append(q_current.copy())
            position_errors.append(position_error)
            orientation_errors.append(orientation_error)

        return RobotResult({
            "targets": normalized_targets,
            "joint_targets": np.asarray(configurations, dtype=float),
            "position_errors": np.asarray(position_errors, dtype=float),
            "orientation_errors": np.asarray(orientation_errors, dtype=float),
        })

    def plan_cartesian_task(
        self,
        targets,
        q_start,
        *,
        steps_per_segment=10,
        ik_kwargs=None,
        position_tolerance=1e-4,
        orientation_tolerance=1e-3,
        check_collisions=True,
        check_self_collision=False,
        link_radius=0.0,
        collision_step=0.05,
        clearance=0.0,
    ):
        """Plan a task with continuous Cartesian interpolation between targets.

        Position-only targets are linearly interpolated.  Homogeneous targets
        use linear position interpolation and quaternion SLERP for orientation.
        Every interpolated target is solved with the previous configuration as
        the IK seed, so the returned joint path follows the Cartesian task
        instead of only matching its sparse waypoints.
        """

        q_start = _joint_vector(q_start, self.n_joints, "q_start")
        self._check_joint_limits(q_start, "q_start")
        targets = list(targets)
        if not targets:
            raise ValueError("targets must be non-empty")
        steps_per_segment = int(steps_per_segment)
        if steps_per_segment < 1:
            raise ValueError("steps_per_segment must be at least 1")
        clearance = float(clearance)
        if clearance < 0.0 or not np.isfinite(clearance):
            raise ValueError("clearance must be a non-negative finite value")

        normalized = []
        for target in targets:
            target = np.asarray(target, dtype=float)
            if target.shape not in {(3,), (4, 4)} or not np.all(np.isfinite(target)):
                raise ValueError("each target must be a finite 3-vector or 4x4 transform")
            normalized.append(target.copy())

        current_pose = self.robot.fk(q_start)
        interpolated = []
        for target in normalized:
            if target.shape == (3,):
                start_position = current_pose[:3, 3]
                for alpha in np.linspace(0.0, 1.0, steps_per_segment + 1)[1:]:
                    interpolated.append(start_position + alpha * (target - start_position))
            else:
                alphas = np.linspace(0.0, 1.0, steps_per_segment + 1)[1:]
                interpolated.extend(interpolate_pose_batch(current_pose, target, alphas))
            current_pose = target if target.shape == (4, 4) else homogeneous_transform(current_pose[:3, :3], target)

        ik_result = self.solve_tcp_path(
            interpolated,
            q_start,
            ik_kwargs=ik_kwargs,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            check_collisions=check_collisions,
            check_self_collision=check_self_collision,
            link_radius=link_radius,
        )
        joint_path = np.vstack([q_start, ik_result["joint_targets"]])
        if check_collisions and not self._collision_free_path(
            joint_path,
            collision_step=collision_step,
            clearance=clearance,
            link_radius=link_radius,
            check_self_collision=check_self_collision,
        ):
            raise RuntimeError("interpolated Cartesian path does not satisfy collision clearance")
        return RobotResult({
            "targets": normalized,
            "interpolated_targets": interpolated,
            "ik": ik_result,
            "joint_path": joint_path,
        })

    def plan_joint_path(
        self,
        q_start,
        q_goal,
        bounds=None,
        *,
        planner="rrt",
        smooth_path=True,
        shortcut_attempts=100,
        collision_step=0.05,
        clearance=0.0,
        link_radius=0.0,
        check_self_collision=False,
        workers=1,
        **rrt_kwargs,
    ):
        """Return a collision-free joint-space path, using a direct path first."""

        q_start = _joint_vector(q_start, self.n_joints, "q_start")
        q_goal = _joint_vector(q_goal, self.n_joints, "q_goal")
        self._check_joint_limits(q_start, "q_start")
        self._check_joint_limits(q_goal, "q_goal")
        bounds = self._resolve_bounds(bounds)
        if np.any(q_start < bounds[:, 0]) or np.any(q_start > bounds[:, 1]):
            raise ValueError("q_start must be inside bounds")
        if np.any(q_goal < bounds[:, 0]) or np.any(q_goal > bounds[:, 1]):
            raise ValueError("q_goal must be inside bounds")

        direct = np.vstack([q_start, q_goal])
        if self._collision_free_path(
            direct,
            collision_step=collision_step,
            clearance=clearance,
            link_radius=link_radius,
            check_self_collision=check_self_collision,
        ):
            return direct

        clearance = float(clearance)
        if clearance < 0.0 or not np.isfinite(clearance):
            raise ValueError("clearance must be a non-negative finite value")
        workers = int(workers)
        if workers < 1:
            raise ValueError("workers must be at least 1")
        def hit(values):
            report = self._state_collision_report(
                values,
                link_radius=link_radius,
                check_self_collision=check_self_collision,
            )
            return bool(report["collision"] or report["minimum_clearance"] < clearance)

        def edge_free(start, end, step):
            return self._collision_free_edge(
                start,
                end,
                step,
                clearance,
                link_radius=link_radius,
                check_self_collision=check_self_collision,
            )

        planner_kwargs = dict(rrt_kwargs)
        planner_kwargs.setdefault("collision_step", collision_step)
        planner_kwargs.setdefault("edge_free", edge_free)
        planner_kwargs.setdefault("workers", workers)
        planner = str(planner).lower()
        if planner == "rrt":
            path = self.world.plan_path(q_start, q_goal, bounds, hit=hit, **planner_kwargs)
        elif planner in {"rrt_star", "rrt*"}:
            path = rrt_star(q_start, q_goal, bounds, hit=hit, **planner_kwargs)
        else:
            raise ValueError("planner must be one of: rrt, rrt_star")
        if path is None:
            raise RuntimeError(f"{planner} could not find a collision-free joint-space path")
        if smooth_path:
            candidate = smooth(
                path,
                hit=hit,
                n=int(shortcut_attempts),
                random_state=rrt_kwargs.get("random_state"),
                edge_free=edge_free,
                workers=workers,
            )
            if self._collision_free_path(
                candidate,
                collision_step=collision_step,
                clearance=clearance,
                link_radius=link_radius,
                check_self_collision=check_self_collision,
            ):
                path = candidate
        if not self._collision_free_path(
            path,
            collision_step=collision_step,
            clearance=clearance,
            link_radius=link_radius,
            check_self_collision=check_self_collision,
        ):
            raise RuntimeError("planner returned a path that does not satisfy collision checks")
        return path

    def optimize_joint_path(
        self,
        joint_path,
        bounds=None,
        *,
        smoothness=1.0,
        reference_weight=1.0,
        collision_weight=1.0,
        max_iter=100,
        step_size=0.1,
        tolerance=1e-6,
        finite_difference_eps=1e-5,
        clearance=0.0,
        link_radius=0.0,
        collision_step=0.05,
        check_self_collision=False,
        velocity_limits=None,
        acceleration_limits=None,
        jerk_limits=None,
        segment_durations=None,
        num_samples_per_segment=101,
        minimum_duration=1e-3,
    ):
        """Optimize a joint path while retaining a collision-safe contract.

        The optimizer minimizes path curvature and a reference-path penalty,
        with a finite-difference clearance barrier from the workcell. Start
        and goal configurations remain fixed. The returned result includes the
        optimized path and a final collision summary; callers should check
        ``result["collision_free"]`` before executing it.
        """

        path = np.asarray(joint_path, dtype=float)
        if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != self.n_joints:
            raise ValueError("joint_path must have shape (n_points, n_joints) with at least two points")
        if not np.all(np.isfinite(path)):
            raise ValueError("joint_path must contain only finite values")
        bounds = self._resolve_bounds(bounds)
        if np.any(path < bounds[:, 0]) or np.any(path > bounds[:, 1]):
            raise ValueError("joint_path exceeds bounds")
        clearance = float(clearance)
        link_radius = float(link_radius)
        collision_step = float(collision_step)
        if clearance < 0.0 or link_radius < 0.0 or collision_step <= 0.0:
            raise ValueError("clearance and link_radius must be non-negative; collision_step must be positive")

        def collision_cost(candidate):
            summary = self.path_collision_summary(
                candidate,
                link_radius=link_radius,
                interpolation_step=collision_step,
                check_self_collision=check_self_collision,
            )
            clearances = np.asarray(summary["clearances"], dtype=float)
            finite = clearances[np.isfinite(clearances)]
            if finite.size == 0:
                return 0.0
            deficit = np.maximum(clearance - finite, 0.0)
            penetration = np.maximum(-finite, 0.0)
            return float(np.mean(deficit**2 + 10.0 * penetration**2))

        result = _optimize_joint_path(
            path,
            smoothness=smoothness,
            reference_weight=reference_weight,
            path_cost=collision_cost,
            path_cost_weight=collision_weight,
            joint_limits=bounds,
            max_iter=max_iter,
            step_size=step_size,
            tolerance=tolerance,
            finite_difference_eps=finite_difference_eps,
            velocity_limits=velocity_limits,
            acceleration_limits=acceleration_limits,
            jerk_limits=jerk_limits,
            segment_durations=segment_durations,
            num_samples_per_segment=num_samples_per_segment,
            minimum_duration=minimum_duration,
        )
        optimized = np.asarray(result["path"], dtype=float)
        collision = self.path_collision_summary(
            optimized,
            link_radius=link_radius,
            interpolation_step=collision_step,
            check_self_collision=check_self_collision,
        )
        result["collision_free"] = bool(
            not collision["collision"] and collision["minimum_clearance"] >= clearance
        )
        result["minimum_clearance"] = collision["minimum_clearance"]
        result["collision_summary"] = collision
        result["bounds"] = bounds
        result["clearance"] = clearance
        if any(value is not None for value in (velocity_limits, acceleration_limits, jerk_limits)):
            trajectory = self.retime_joint_path(
                optimized,
                velocity_limits=velocity_limits
                if velocity_limits is not None
                else np.full(self.n_joints, 1e12),
                acceleration_limits=acceleration_limits,
                jerk_limits=jerk_limits,
                num_samples_per_segment=num_samples_per_segment,
                minimum_duration=minimum_duration,
            )
            result["trajectory"] = trajectory
            checks = []
            for key, limit in (
                ("velocity", velocity_limits),
                ("acceleration", acceleration_limits),
                ("jerk", jerk_limits),
            ):
                if limit is not None:
                    checks.append(
                        np.all(
                            np.max(np.abs(trajectory[key]), axis=0)
                            <= np.asarray(limit).reshape(-1) + 1e-10
                        )
                    )
            result["trajectory_limits_feasible"] = bool(all(checks))
        return result

    def plan_motion(
        self,
        q_start,
        q_goal,
        bounds=None,
        *,
        planner="rrt",
        velocity_limits=1.0,
        acceleration_limits=None,
        jerk_limits=None,
        clearance=0.0,
        collision_step=0.05,
        link_radius=0.0,
        check_self_collision=False,
        smooth_path=True,
        shortcut_attempts=100,
        workers=1,
        retime_kwargs=None,
        track=False,
        tracking_kwargs=None,
        optimize_path=False,
        optimization_kwargs=None,
        **planner_kwargs,
    ):
        """Plan, retime, and diagnose a joint-space motion in one call."""

        path = self.plan_joint_path(
            q_start,
            q_goal,
            bounds,
            planner=planner,
            smooth_path=smooth_path,
            shortcut_attempts=shortcut_attempts,
            collision_step=collision_step,
            clearance=clearance,
            link_radius=link_radius,
            check_self_collision=check_self_collision,
            workers=workers,
            **planner_kwargs,
        )
        optimization = None
        if optimize_path:
            optimization_kwargs = {} if optimization_kwargs is None else dict(optimization_kwargs)
            optimization_kwargs.setdefault("clearance", clearance)
            optimization_kwargs.setdefault("collision_step", collision_step)
            optimization_kwargs.setdefault("check_self_collision", check_self_collision)
            optimization = self.optimize_joint_path(
                path,
                bounds=bounds,
                **optimization_kwargs,
            )
            if not optimization["collision_free"]:
                raise RuntimeError("trajectory optimizer did not produce a collision-safe path")
            path = np.asarray(optimization["path"], dtype=float)
        collision = self.path_collision_report(
            path,
            link_radius=link_radius,
            interpolation_step=collision_step,
            check_self_collision=check_self_collision,
        )
        deltas = np.diff(path, axis=0)
        planning_report = RobotResult({
            "planner": str(planner),
            "n_waypoints": int(path.shape[0]),
            "joint_path_length": float(np.sum(np.linalg.norm(deltas, axis=1))) if deltas.size else 0.0,
            "minimum_clearance": collision["minimum_clearance"],
            "collision": collision["collision"],
            "clearance_margin": float(clearance),
            "clearance_violation": bool(collision["minimum_clearance"] < float(clearance)),
            "collision_samples": collision["n_samples"],
            "optimized": optimization is not None,
            "optimization_final_cost": None
            if optimization is None
            else float(optimization["final_cost"]),
        })

        retime_kwargs = {} if retime_kwargs is None else dict(retime_kwargs)
        trajectory = self.retime_joint_path(
            path,
            velocity_limits=velocity_limits,
            acceleration_limits=acceleration_limits,
            jerk_limits=jerk_limits,
            **retime_kwargs,
        )
        tracking = None
        if track:
            tracking_kwargs = {} if tracking_kwargs is None else dict(tracking_kwargs)
            tracking = self.simulate_tracking(trajectory, **tracking_kwargs)
        safety = self.validate_trajectory(
            trajectory,
            tracking=tracking,
            clearance_margin=clearance,
            link_radius=link_radius,
            check_self_collision=check_self_collision,
            raise_on_failure=True,
        )
        result = RobotResult({
            "path": path,
            "trajectory": trajectory,
            "tracking": tracking,
            "optimization": optimization,
            "planning_report": planning_report,
            "collision_report": collision,
            "safety_report": safety,
        })
        self._store_report(safety)
        self._store_artifact("motion", result)
        return result

    def plan_tcp_task(self, targets, q_start, bounds=None, *, ik_kwargs=None, **planning_kwargs):
        """Plan a complete TCP task from continuous IK through collision-free motion."""

        ik_result = self.solve_tcp_path(targets, q_start, ik_kwargs=ik_kwargs)
        current = _joint_vector(q_start, self.n_joints, "q_start")
        segments: list[np.ndarray] = []
        for goal in ik_result["joint_targets"]:
            segment = self.plan_joint_path(current, goal, bounds, **planning_kwargs)
            segments.append(segment if not segments else segment[1:])
            current = goal
        path = np.concatenate(segments, axis=0)
        return RobotResult({"ik": ik_result, "joint_path": path})

    def plan_pick_and_place(
        self,
        pick_target,
        place_target,
        q_start,
        bounds=None,
        *,
        approach_offset,
        retreat_offset=None,
        steps_per_segment=10,
        ik_kwargs=None,
        position_tolerance=1e-4,
        orientation_tolerance=1e-3,
        planner="rrt",
        smooth_path=True,
        shortcut_attempts=100,
        collision_step=0.05,
        clearance=0.0,
        link_radius=0.0,
        check_self_collision=False,
        workers=1,
        velocity_limits=1.0,
        acceleration_limits=None,
        jerk_limits=None,
        retime_kwargs=None,
        gripper_open=1.0,
        gripper_closed=0.0,
        **planner_kwargs,
    ):
        """Plan a collision-aware pick-and-place cycle with gripper events.

        ``approach_offset`` and ``retreat_offset`` are world-frame vectors
        from the grasp or release TCP target.  The robot follows Cartesian
        paths while approaching and retracting; the loaded transfer between
        the two safe approach poses uses the configured joint-space planner.
        The returned trajectory has a time-aligned ``gripper_schedule`` for
        execution adapters.
        """

        pick_target = np.asarray(pick_target, dtype=float)
        place_target = np.asarray(place_target, dtype=float)
        if pick_target.shape not in {(3,), (4, 4)} or not np.all(np.isfinite(pick_target)):
            raise ValueError("pick_target must be a finite 3-vector or 4x4 transform")
        if place_target.shape not in {(3,), (4, 4)} or not np.all(np.isfinite(place_target)):
            raise ValueError("place_target must be a finite 3-vector or 4x4 transform")
        if pick_target.shape != place_target.shape:
            raise ValueError("pick_target and place_target must have the same shape")
        approach_offset = _task_offset(approach_offset, "approach_offset")
        if retreat_offset is None:
            retreat_offset = approach_offset
        else:
            retreat_offset = _task_offset(retreat_offset, "retreat_offset")
        gripper_open = float(gripper_open)
        gripper_closed = float(gripper_closed)
        if not np.isfinite(gripper_open) or not np.isfinite(gripper_closed):
            raise ValueError("gripper values must be finite")

        pick_approach = _tcp_offset(pick_target, approach_offset, "pick_target")
        pick_retreat = _tcp_offset(pick_target, retreat_offset, "pick_target")
        place_approach = _tcp_offset(place_target, approach_offset, "place_target")
        place_retreat = _tcp_offset(place_target, retreat_offset, "place_target")
        cartesian_kwargs = {
            "steps_per_segment": steps_per_segment,
            "ik_kwargs": ik_kwargs,
            "position_tolerance": position_tolerance,
            "orientation_tolerance": orientation_tolerance,
            "collision_step": collision_step,
            "clearance": clearance,
            "link_radius": link_radius,
            "check_self_collision": check_self_collision,
        }
        pick_task = self.plan_cartesian_task(
            [pick_approach, pick_target, pick_retreat], q_start, **cartesian_kwargs
        )
        place_approach_ik = self.solve_tcp_path(
            [place_approach],
            pick_task["joint_path"][-1],
            ik_kwargs=ik_kwargs,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            check_self_collision=check_self_collision,
            link_radius=link_radius,
        )
        transfer_path = self.plan_joint_path(
            pick_task["joint_path"][-1],
            place_approach_ik["joint_targets"][-1],
            bounds,
            planner=planner,
            smooth_path=smooth_path,
            shortcut_attempts=shortcut_attempts,
            collision_step=collision_step,
            clearance=clearance,
            link_radius=link_radius,
            check_self_collision=check_self_collision,
            workers=workers,
            **planner_kwargs,
        )
        place_task = self.plan_cartesian_task(
            [place_target, place_retreat], transfer_path[-1], **cartesian_kwargs
        )
        joint_path = np.concatenate(
            [pick_task["joint_path"], transfer_path[1:], place_task["joint_path"][1:]], axis=0
        )
        retime_kwargs = {} if retime_kwargs is None else dict(retime_kwargs)
        trajectory = self.retime_joint_path(
            joint_path,
            velocity_limits=velocity_limits,
            acceleration_limits=acceleration_limits,
            jerk_limits=jerk_limits,
            **retime_kwargs,
        )

        steps = int(steps_per_segment)
        pick_approach_index = steps
        pick_grasp_index = 2 * steps
        pick_retreat_index = 3 * steps
        place_approach_index = pick_task["joint_path"].shape[0] - 1 + transfer_path.shape[0] - 1
        place_release_index = place_approach_index + steps
        place_retreat_index = place_release_index + steps
        waypoint_time = np.concatenate([[0.0], np.cumsum(trajectory["durations"])])
        phase_indices = {
            "pick_approach": pick_approach_index,
            "pick_grasp": pick_grasp_index,
            "pick_retreat": pick_retreat_index,
            "place_approach": place_approach_index,
            "place_release": place_release_index,
            "place_retreat": place_retreat_index,
        }
        gripper_schedule = [
            {"time": 0.0, "path_index": 0, "command": "open", "value": gripper_open},
            {
                "time": float(waypoint_time[pick_grasp_index]),
                "path_index": pick_grasp_index,
                "command": "close",
                "value": gripper_closed,
            },
            {
                "time": float(waypoint_time[place_release_index]),
                "path_index": place_release_index,
                "command": "open",
                "value": gripper_open,
            },
        ]
        safety = self.validate_trajectory(
            trajectory,
            clearance_margin=clearance,
            link_radius=link_radius,
            check_self_collision=check_self_collision,
            raise_on_failure=True,
        )
        result = RobotResult({
            "pick_target": pick_target.copy(),
            "place_target": place_target.copy(),
            "approach_offset": approach_offset.copy(),
            "retreat_offset": retreat_offset.copy(),
            "pick_task": pick_task,
            "transfer_path": transfer_path,
            "place_task": place_task,
            "joint_path": joint_path,
            "trajectory": trajectory,
            "phase_indices": phase_indices,
            "gripper_schedule": gripper_schedule,
            "safety_report": safety,
        })
        self._store_report(safety)
        self._store_artifact("pick_and_place", result)
        return result

    def retime_joint_path(
        self,
        joint_path,
        velocity_limits,
        acceleration_limits=None,
        jerk_limits=None,
        *,
        num_samples_per_segment=101,
        minimum_duration=1e-3,
    ):
        """Time-parameterize a path under quintic velocity, acceleration, and jerk limits."""

        path = np.asarray(joint_path, dtype=float)
        if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != self.n_joints:
            raise ValueError("joint_path must have shape (n_points, n_joints) with at least two points")
        if not np.all(np.isfinite(path)):
            raise ValueError("joint_path must contain only finite values")
        self._check_joint_limits(path, "joint_path")
        samples = int(num_samples_per_segment)
        minimum_duration = float(minimum_duration)
        if samples < 2 or minimum_duration <= 0.0:
            raise ValueError("num_samples_per_segment must be at least 2 and minimum_duration must be positive")

        velocity_limits = _limits(velocity_limits, self.n_joints, "velocity_limits")
        if acceleration_limits is not None:
            acceleration_limits = _limits(acceleration_limits, self.n_joints, "acceleration_limits")
        if jerk_limits is not None:
            jerk_limits = _limits(jerk_limits, self.n_joints, "jerk_limits")

        compiled = _retime_quintic_path_compiled(
            path,
            velocity_limits,
            acceleration_limits,
            jerk_limits,
            samples,
            minimum_duration,
        )
        if compiled is None:
            duration_values: list[float] = []
            for delta in np.abs(np.diff(path, axis=0)):
                candidates = [_QUINTIC_MAX_VELOCITY * delta / velocity_limits]
                if acceleration_limits is not None:
                    candidates.append(np.sqrt(_QUINTIC_MAX_ACCELERATION * delta / acceleration_limits))
                if jerk_limits is not None:
                    candidates.append(np.cbrt(_QUINTIC_MAX_JERK * delta / jerk_limits))
                duration_values.append(max(minimum_duration, float(np.max(np.concatenate(candidates)))))
            durations = np.asarray(duration_values, dtype=float)
            time, position, velocity, acceleration = sample_joint_trajectory_segments(
                path,
                durations,
                num_samples_per_segment=samples,
                kind="quintic",
            )

            delta = path[1:] - path[:-1]
            jerk_parts = []
            for segment, duration in enumerate(durations):
                tau = np.linspace(0.0, 1.0, samples)
                if segment > 0:
                    tau = tau[1:]
                jerk_scale = (60.0 - 360.0 * tau + 360.0 * tau**2) / duration**3
                jerk_parts.append(jerk_scale[:, None] * delta[segment])
            jerk = np.concatenate(jerk_parts, axis=0)
        else:
            time, position, velocity, acceleration, jerk, durations = compiled
        return JointTrajectory(
            time=time,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            durations=durations,
            path=path.copy(),
            velocity_limits=velocity_limits.copy(),
            acceleration_limits=None if acceleration_limits is None else acceleration_limits.copy(),
            jerk_limits=None if jerk_limits is None else jerk_limits.copy(),
        )

    def simulate_tracking(
        self,
        trajectory,
        *,
        gains=(4.0, 0.4),
        dt=None,
        damping=0.0,
        state=None,
        controller="pd",
        mpc_kwargs=None,
        control_limits=None,
    ):
        """Track a trajectory with the legacy PD or optional linear MPC controller."""

        if isinstance(trajectory, Mapping):
            reference = np.asarray(trajectory["position"], dtype=float)
            reference_time = np.asarray(trajectory.get("time"), dtype=float)
            if reference_time.shape != (reference.shape[0],):
                raise ValueError("trajectory time must have one entry per reference position")
        else:
            reference = np.asarray(trajectory, dtype=float)
            reference_time = None
        if reference.ndim != 2 or reference.shape[0] < 1 or reference.shape[1] != self.n_joints:
            raise ValueError("trajectory positions must have shape (n_steps, n_joints)")
        if not np.all(np.isfinite(reference)):
            raise ValueError("trajectory positions must contain only finite values")
        self._check_joint_limits(reference, "trajectory positions")

        if dt is None:
            if reference_time is None or reference_time.size < 2:
                dt = 0.1
            else:
                differences = np.diff(reference_time)
                if np.any(differences <= 0.0):
                    raise ValueError("trajectory time must be strictly increasing")
                dt = float(np.min(differences))
        dt = float(dt)
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("dt must be positive and finite")

        if reference_time is not None and reference_time.size > 1:
            simulation_time = np.arange(reference_time[0], reference_time[-1], dt)
            if simulation_time.size == 0 or not np.isclose(simulation_time[-1], reference_time[-1]):
                simulation_time = np.append(simulation_time, reference_time[-1])
            reference = np.column_stack(
                [np.interp(simulation_time, reference_time, reference[:, joint]) for joint in range(self.n_joints)]
            )
        else:
            simulation_time = np.arange(reference.shape[0], dtype=float) * dt

        from ..control import ComputedTorqueController, JointMPCController, JointPDController
        from ..sim.core import SimpleRobotSim

        controller_impl: Any
        if isinstance(controller, str):
            controller_name = controller.lower()
            if controller_name == "pd":
                kp, kd = gains
                output_limits = None
                if control_limits is not None:
                    limits = _limits(control_limits, self.n_joints, "control_limits")
                    output_limits = (-limits, limits)
                controller_impl = JointPDController(
                    self.n_joints, kp=kp, kd=kd, output_limits=output_limits
                )
            elif controller_name == "mpc":
                damping = float(damping)
                if damping < 0.0 or not np.isfinite(damping):
                    raise ValueError("damping must be a non-negative finite value")
                mpc_options = {} if mpc_kwargs is None else dict(mpc_kwargs)
                mpc_options.setdefault("horizon", max(2, min(20, reference.shape[0] - 1)))
                if control_limits is not None and "u_bounds" not in mpc_options:
                    limits = _limits(control_limits, self.n_joints, "control_limits")
                    mpc_options["u_bounds"] = (-limits, limits)
                controller_impl = JointMPCController(
                    self.n_joints,
                    dt=dt,
                    damping=damping,
                    mpc_kwargs=mpc_options,
                )
            elif controller_name in {"computed_torque", "computed-torque"}:
                kp, kd = gains
                output_limits = None
                if control_limits is not None:
                    limits = _limits(control_limits, self.n_joints, "control_limits")
                    output_limits = (-limits, limits)
                controller_impl = ComputedTorqueController(
                    self.robot,
                    kp=kp,
                    kd=kd,
                    output_limits=output_limits,
                )
            else:
                raise ValueError("controller must be 'pd', 'mpc', 'computed_torque', or a Controller")
        else:
            controller_impl = controller
            controller_name = str(getattr(controller_impl, "name", type(controller_impl).__name__)).lower()
            if not callable(getattr(controller_impl, "update", None)):
                raise TypeError("controller objects must define update(reference, state, dt)")
            if not callable(getattr(controller_impl, "reset", None)):
                raise TypeError("controller objects must define reset(state=None)")
            if not callable(getattr(controller_impl, "status", None)):
                raise TypeError("controller objects must define status()")
            if getattr(controller_impl, "n_joints", self.n_joints) != self.n_joints:
                raise ValueError("controller.n_joints must match the workcell robot")

        robot = cast(Any, self.robot)
        sim = SimpleRobotSim(self.robot, state=state, dt=dt, damping=damping)
        controller_impl.reset(state=sim.state.copy())
        state_history = [sim.state.copy()]
        pose_history = [sim.pose()]
        control_history = []
        command_type = str(getattr(controller_impl, "command_type", "acceleration")).lower()
        if command_type not in {"acceleration", "torque"}:
            raise ValueError("controller command_type must be 'acceleration' or 'torque'")
        for target in reference:
            action = _joint_vector(
                controller_impl.update(target, sim.state.copy(), dt),
                self.n_joints,
                "controller output",
            )
            control_history.append(action)
            if command_type == "torque":
                gravity = getattr(controller_impl, "gravity", (0.0, 0.0, -9.81))
                if callable(getattr(robot, "forward_dynamics_batch", None)):
                    acceleration = robot.forward_dynamics_batch(
                        sim.q[None, :],
                        sim.qd[None, :],
                        action[None, :],
                        gravity=gravity,
                        include_coriolis=True,
                    )[0]
                elif callable(getattr(robot, "forward_dynamics", None)):
                    acceleration = robot.forward_dynamics(
                        sim.q,
                        sim.qd,
                        action,
                        gravity=gravity,
                        include_coriolis=True,
                    )
                else:
                    raise TypeError("torque controllers require robot forward dynamics")
                sim.step(acceleration)
            else:
                sim.step(action)
            state_history.append(sim.state.copy())
            pose_history.append(sim.pose())
        states = np.asarray(state_history)
        controls = np.asarray(control_history)
        controller_status = controller_impl.status()
        actual = states[1:, : self.n_joints]
        joint_error = actual - reference
        return RobotResult({
            "time": simulation_time,
            "reference": reference,
            "states": states,
            "poses": pose_history,
            "controls": controls,
            "actual": actual,
            "joint_error": joint_error,
            "dt": dt,
            "controller": controller_name,
            "controller_status": controller_status,
        })

    def safety_report(
        self,
        trajectory,
        tracking=None,
        *,
        clearance_margin=0.0,
        singularity_threshold=1e-8,
        link_radius=0.0,
        interpolation_step=0.05,
        check_self_collision=False,
    ):
        """Summarize collision clearance, motion limits, and tracking error."""

        if isinstance(trajectory, Mapping):
            position = np.asarray(trajectory["position"], dtype=float)
            raw_time = trajectory.get("time")
            time = None if raw_time is None else np.asarray(raw_time, dtype=float)
            velocity = trajectory.get("velocity")
            acceleration = trajectory.get("acceleration")
            jerk = trajectory.get("jerk")
        else:
            position = np.asarray(trajectory, dtype=float)
            time = None
            velocity = acceleration = jerk = None
        clearance_margin = float(clearance_margin)
        if clearance_margin < 0.0 or not np.isfinite(clearance_margin):
            raise ValueError("clearance_margin must be a non-negative finite value")
        if position.ndim != 2 or position.shape[0] < 1 or position.shape[1] != self.n_joints:
            raise ValueError("trajectory positions must have shape (n_steps, n_joints)")
        link_radius = float(link_radius)
        interpolation_step = float(interpolation_step)
        if link_radius < 0.0 or not np.isfinite(link_radius):
            raise ValueError("link_radius must be a non-negative finite value")
        if interpolation_step <= 0.0 or not np.isfinite(interpolation_step):
            raise ValueError("interpolation_step must be a positive finite value")
        time_valid = time is None or (
            time.shape == (position.shape[0],)
            and np.all(np.isfinite(time))
            and np.all(np.diff(time) > 0.0)
        )
        if time is not None and time.shape != (position.shape[0],):
            raise ValueError("trajectory time must have one value per position sample")

        reference_collision = self.path_collision_summary(
            position,
            link_radius=link_radius,
            interpolation_step=interpolation_step,
            check_self_collision=check_self_collision,
        )
        reference_clearances = np.asarray(reference_collision["clearances"], dtype=float)
        deltas = np.diff(position, axis=0)
        joint_path_length = float(np.sum(np.linalg.norm(deltas, axis=1))) if deltas.size else 0.0
        reference_limit_violation = self._joint_limit_violation(position)
        tracking_clearances = None
        tracking_limit_violation = None
        actual = None
        if tracking is not None and "actual" in tracking:
            actual = np.asarray(tracking["actual"], dtype=float)
            if actual.ndim != 2 or actual.shape[1] != self.n_joints or not np.all(np.isfinite(actual)):
                raise ValueError("tracking actual must have shape (n_steps, n_joints) with finite values")
            tracking_collision = self.path_collision_summary(
                actual,
                link_radius=link_radius,
                interpolation_step=interpolation_step,
                check_self_collision=check_self_collision,
            )
            tracking_clearances = np.asarray(tracking_collision["clearances"], dtype=float)
            tracking_limit_violation = self._joint_limit_violation(actual)

        all_clearances = reference_clearances if tracking_clearances is None else np.concatenate([reference_clearances, tracking_clearances])
        finite_clearances = all_clearances[np.isfinite(all_clearances)]
        limit_violation = reference_limit_violation
        if tracking_limit_violation is not None:
            limit_violation = np.concatenate([limit_violation, tracking_limit_violation])
        motion_limits = {
            "velocity": None if isinstance(trajectory, np.ndarray) else trajectory.get("velocity_limits"),
            "acceleration": None if isinstance(trajectory, np.ndarray) else trajectory.get("acceleration_limits"),
            "jerk": None if isinstance(trajectory, np.ndarray) else trajectory.get("jerk_limits"),
        }
        motion_metrics: dict[str, dict[str, object] | None] = {}
        motion_values = {}
        for name, values, limits in (
            ("velocity", velocity, motion_limits["velocity"]),
            ("acceleration", acceleration, motion_limits["acceleration"]),
            ("jerk", jerk, motion_limits["jerk"]),
        ):
            if values is None:
                motion_metrics[name] = None
                continue
            values = np.asarray(values, dtype=float)
            if values.shape != position.shape:
                raise ValueError(f"trajectory {name} must have shape {position.shape}")
            motion_values[name] = values

        motion_maxima = {}
        if motion_values:
            names = list(motion_values)
            stacked = np.stack([motion_values[name] for name in names], axis=2)
            maxima = trajectory_peaks_batch(stacked)
            motion_maxima = {name: maxima[index] for index, name in enumerate(names)}
        for name, values, limits in (
            (name, motion_values.get(name), motion_limits[name])
            for name in ("velocity", "acceleration", "jerk")
        ):
            if values is None:
                motion_metrics[name] = None
                continue
            maximum = motion_maxima[name]
            violation = None if limits is None else bool(np.any(maximum > np.asarray(limits) + 1e-12))
            motion_metrics[name] = {
                "maximum_by_joint": maximum.tolist(),
                "limits": None if limits is None else np.asarray(limits, dtype=float).tolist(),
                "violation": violation,
            }
        kinematics = self.robot.kinematic_metrics_batch(
            position, translational=True, threshold=singularity_threshold
        )
        condition_numbers = np.asarray(kinematics["condition_number"], dtype=float)
        manipulabilities = np.asarray(kinematics["manipulability"], dtype=float)
        clearance_violation = bool(np.any(all_clearances < clearance_margin))
        motion_violation = any(
            item is not None and item["violation"] is True for item in motion_metrics.values()
        )
        report = RobotResult({
            "workcell": self.name,
            "n_joints": self.n_joints,
            "n_samples": int(position.shape[0]),
            "duration": None if time is None else float(time[-1] - time[0]),
            "time_valid": bool(time_valid),
            "joint_path_length": joint_path_length,
            "collision": bool(np.any(all_clearances <= 0.0)),
            "clearance_margin": clearance_margin,
            "link_radius": link_radius,
            "interpolation_step": interpolation_step,
            "check_self_collision": bool(check_self_collision),
            "clearance_violation": clearance_violation,
            "self_collision": bool(reference_collision.get("self_collision", False))
            or bool(
                tracking_collision.get("self_collision", False)
                if tracking_clearances is not None
                else False
            ),
            "reference_collision": bool(np.any(reference_clearances <= 0.0)),
            "reference_first_collision_index": reference_collision["first_collision_index"],
            "tracking_collision": None if tracking_clearances is None else bool(np.any(tracking_clearances <= 0.0)),
            "tracking_first_collision_index": None
            if tracking_clearances is None
            else tracking_collision["first_collision_index"],
            "minimum_clearance": None if finite_clearances.size == 0 else float(np.min(finite_clearances)),
            "reference_minimum_clearance": None
            if not np.any(np.isfinite(reference_clearances))
            else float(np.min(reference_clearances[np.isfinite(reference_clearances)])),
            "tracking_minimum_clearance": None
            if tracking_clearances is None or not np.any(np.isfinite(tracking_clearances))
            else float(np.min(tracking_clearances[np.isfinite(tracking_clearances)])),
            "joint_limits": None if self.joint_limits is None else self.joint_limits.tolist(),
            "joint_limit_violation": bool(np.any(limit_violation > 0.0)),
            "maximum_joint_limit_violation": None
            if self.joint_limits is None
            else float(np.max(limit_violation)),
            "max_velocity": None if velocity is None else float(np.max(motion_maxima["velocity"])),
            "max_acceleration": None
            if acceleration is None
            else float(np.max(motion_maxima["acceleration"])),
            "max_jerk": None if jerk is None else float(np.max(motion_maxima["jerk"])),
            "motion_limits": motion_metrics,
            "motion_limit_violation": motion_violation,
            "minimum_position_manipulability": float(np.min(manipulabilities)),
            "maximum_position_condition_number": float(np.max(condition_numbers)),
            "singular_configuration": bool(np.any(kinematics["singular"])),
        })
        if tracking is not None:
            error = np.asarray(tracking["joint_error"], dtype=float)
            if error.ndim != 2 or error.shape[1] != self.n_joints:
                raise ValueError("tracking joint_error must have shape (n_steps, n_joints)")
            error_norm = np.linalg.norm(error, axis=1)
            report["tracking_max_error"] = float(np.max(error_norm))
            report["tracking_rms_error"] = float(np.sqrt(np.mean(error_norm**2)))
        return report

    def validate_trajectory(
        self,
        trajectory,
        tracking=None,
        *,
        clearance_margin=0.0,
        singularity_threshold=1e-8,
        link_radius=0.0,
        interpolation_step=0.05,
        check_limits=True,
        check_collision=True,
        check_clearance=True,
        check_motion_limits=True,
        check_singularity=False,
        check_time=True,
        check_self_collision=False,
        raise_on_failure=False,
    ):
        """Validate a trajectory before handing it to an execution adapter.

        The returned report remains mapping-compatible with ``safety_report``
        and adds a stable ``valid`` flag plus machine-readable violations.
        """

        report = self.safety_report(
            trajectory,
            tracking=tracking,
            clearance_margin=clearance_margin,
            singularity_threshold=singularity_threshold,
            link_radius=link_radius,
            interpolation_step=interpolation_step,
            check_self_collision=check_self_collision,
        )
        violations = []
        if check_time and not report["time_valid"]:
            violations.append("invalid_time")
        if check_limits and report["joint_limit_violation"]:
            violations.append("joint_limits")
        if check_collision and report["collision"]:
            violations.append("collision")
        if check_self_collision and report["self_collision"]:
            violations.append("self_collision")
        if check_clearance and report["clearance_violation"]:
            violations.append("clearance")
        if check_motion_limits and report["motion_limit_violation"]:
            violations.append("motion_limits")
        if check_singularity and report["singular_configuration"]:
            violations.append("singularity")
        report["violations"] = violations
        report["valid"] = not violations
        report["execution_ready"] = report["valid"]
        candidates = []
        if isinstance(trajectory, Mapping):
            positions = np.asarray(trajectory["position"], dtype=float)
            if check_limits:
                invalid = np.flatnonzero(self._joint_limit_violation(positions) > 0.0)
                if invalid.size:
                    candidates.append(int(invalid[0]))
            if check_motion_limits:
                for name in ("velocity", "acceleration", "jerk"):
                    values = trajectory.get(name)
                    limits = trajectory.get(f"{name}_limits")
                    if values is None or limits is None:
                        continue
                    values = np.asarray(values, dtype=float)
                    limits = np.asarray(limits, dtype=float).reshape(-1)
                    invalid = np.flatnonzero(np.any(np.abs(values) > limits + 1e-12, axis=1))
                    if invalid.size:
                        candidates.append(int(invalid[0]))
        if report["reference_first_collision_index"] is not None:
            candidates.append(report["reference_first_collision_index"])
        if report["tracking_first_collision_index"] is not None:
            candidates.append(report["tracking_first_collision_index"])
        report["first_violation_index"] = min(candidates) if candidates else None
        self._store_report(report)
        if raise_on_failure and not report["valid"]:
            raise RuntimeError("trajectory failed safety validation: " + ", ".join(violations))
        return report

    def export_artifacts(self, directory, trajectory, *, tracking=None, report=None):
        """Export the planned trajectory, optional tracking trace, and safety report."""

        if not isinstance(trajectory, Mapping):
            raise TypeError("trajectory must be a mapping returned by retime_joint_path")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        fields = ["time"]
        columns = [np.asarray(trajectory["time"], dtype=float)]
        for prefix, key in (("q", "position"), ("qd", "velocity"), ("qdd", "acceleration"), ("jerk", "jerk")):
            values = np.asarray(trajectory[key], dtype=float)
            fields.extend(f"{prefix}_{joint}" for joint in range(self.n_joints))
            columns.extend(values[:, joint] for joint in range(self.n_joints))
        trajectory_path = directory / "trajectory.csv"
        self._write_csv(trajectory_path, fields, zip(*columns))

        paths = RobotResult({"trajectory_csv": trajectory_path})
        if tracking is not None:
            tracking_path = directory / "tracking.csv"
            fields = ["time"]
            columns = [np.asarray(tracking["time"], dtype=float)]
            for prefix, key in (("q_ref", "reference"), ("q_actual", "actual"), ("q_error", "joint_error")):
                values = np.asarray(tracking[key], dtype=float)
                fields.extend(f"{prefix}_{joint}" for joint in range(self.n_joints))
                columns.extend(values[:, joint] for joint in range(self.n_joints))
            self._write_csv(tracking_path, fields, zip(*columns))
            paths["tracking_csv"] = tracking_path

        report_path = directory / "safety_report.json"
        report = self.safety_report(trajectory, tracking=tracking) if report is None else report
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        paths["safety_report_json"] = report_path
        return paths

    @staticmethod
    def _write_csv(path, fields, rows):
        with Path(path).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)


__all__ = ["RobotWorkcell"]
