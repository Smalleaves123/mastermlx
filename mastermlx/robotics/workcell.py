"""Offline programming workflow for serial manipulators in a planar workcell."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..planning import smooth
from .model import RobotModel
from .results import JointTrajectory, RobotResult
from .trajectory import sample_joint_trajectory_segments
from .transforms import homogeneous_transform, matrix_to_quaternion, quaternion_to_matrix


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


def _joint_limits(values, n_joints):
    if values is None:
        return None
    values = np.asarray(values, dtype=float)
    if values.shape != (n_joints, 2):
        raise ValueError("joint_limits must have shape (n_joints, 2)")
    if not np.all(np.isfinite(values)) or np.any(values[:, 0] >= values[:, 1]):
        raise ValueError("joint_limits must contain finite lower < upper bounds")
    return values.copy()


def _orientation_error(actual, target):
    rotation = target[:3, :3] @ actual[:3, :3].T
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _slerp(first, second, alpha):
    first = np.asarray(first, dtype=float).reshape(4)
    second = np.asarray(second, dtype=float).reshape(4)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    if dot > 0.9995:
        return (first + float(alpha) * (second - first)) / np.linalg.norm(first + float(alpha) * (second - first))
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    weights = np.sin((1.0 - float(alpha)) * angle), np.sin(float(alpha) * angle)
    return (weights[0] * first + weights[1] * second) / np.sin(angle)


class RobotWorkcell:
    """Compose kinematics, collision planning, retiming, and virtual tracking.

    The workcell uses :class:`~mastermlx.sim.SimpleWorld`, so obstacle checks
    model planar circular obstacles.  It deliberately reuses the project's
    robot model and simulator instead of maintaining a second implementation.
    """

    def __init__(self, robot, world=None, name=None, joint_limits=None):
        from ..sim.world import SimpleWorld

        if not isinstance(robot, RobotModel):
            raise TypeError("robot must be a RobotModel")
        if world is None:
            world = SimpleWorld(robot)
        if not isinstance(world, SimpleWorld):
            raise TypeError("world must be a SimpleWorld")
        if world.robot is not robot:
            raise ValueError("world.robot must be the same RobotModel as robot")
        self.robot = robot
        self.world = world
        self.name = robot.name if name is None else str(name)
        self.joint_limits = _joint_limits(joint_limits, len(robot.links))

    @property
    def n_joints(self):
        return len(self.robot.links)

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

    def _collision_free_path(self, path, collision_step=0.05, clearance=0.0):
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[0] < 1 or path.shape[1] != self.n_joints:
            raise ValueError("path must have shape (n_points, n_joints)")
        collision_step = float(collision_step)
        clearance = float(clearance)
        if collision_step <= 0.0 or not np.isfinite(collision_step):
            raise ValueError("collision_step must be a positive finite value")
        if clearance < 0.0 or not np.isfinite(clearance):
            raise ValueError("clearance must be a non-negative finite value")
        for start, end in zip(path[:-1], path[1:]):
            count = max(1, int(np.ceil(np.linalg.norm(end - start) / collision_step)))
            for alpha in np.linspace(0.0, 1.0, count + 1):
                if self.world.clearance(start + alpha * (end - start)) < clearance:
                    return False
        return self.world.clearance(path[-1]) >= clearance

    def solve_tcp_path(
        self,
        targets,
        q_start,
        *,
        ik_kwargs=None,
        position_tolerance=1e-4,
        orientation_tolerance=1e-3,
        check_collisions=True,
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
            if check_collisions and self.world.hit(q_current):
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
                start_position = current_pose[:3, 3]
                start_quaternion = matrix_to_quaternion(current_pose[:3, :3])
                target_quaternion = matrix_to_quaternion(target[:3, :3])
                for alpha in np.linspace(0.0, 1.0, steps_per_segment + 1)[1:]:
                    pose = homogeneous_transform(
                        quaternion_to_matrix(_slerp(start_quaternion, target_quaternion, alpha)),
                        start_position + alpha * (target[:3, 3] - start_position),
                    )
                    interpolated.append(pose)
            current_pose = target if target.shape == (4, 4) else homogeneous_transform(current_pose[:3, :3], target)

        ik_result = self.solve_tcp_path(
            interpolated,
            q_start,
            ik_kwargs=ik_kwargs,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            check_collisions=check_collisions,
        )
        joint_path = np.vstack([q_start, ik_result["joint_targets"]])
        if check_collisions and not self._collision_free_path(
            joint_path, collision_step=collision_step, clearance=clearance
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
        smooth_path=True,
        shortcut_attempts=100,
        collision_step=0.05,
        clearance=0.0,
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
        if self._collision_free_path(direct, collision_step=collision_step, clearance=clearance):
            return direct

        clearance = float(clearance)
        if clearance < 0.0 or not np.isfinite(clearance):
            raise ValueError("clearance must be a non-negative finite value")
        def hit(values):
            return self.world.clearance(values) < clearance

        path = self.world.plan_path(q_start, q_goal, bounds, hit=hit, **rrt_kwargs)
        if path is None:
            raise RuntimeError("RRT could not find a collision-free joint-space path")
        if smooth_path:
            candidate = smooth(
                path,
                hit=hit,
                n=int(shortcut_attempts),
                random_state=rrt_kwargs.get("random_state"),
            )
            if self._collision_free_path(candidate, collision_step=collision_step, clearance=clearance):
                path = candidate
        if not self._collision_free_path(path, collision_step=collision_step, clearance=clearance):
            raise RuntimeError("planner returned a path that does not satisfy collision checks")
        return path

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

        starts = np.concatenate([[0.0], np.cumsum(durations[:-1])])
        segment_indices = np.searchsorted(starts, time, side="right") - 1
        segment_indices = np.clip(segment_indices, 0, durations.size - 1)
        tau = (time - starts[segment_indices]) / durations[segment_indices]
        tau = np.clip(tau, 0.0, 1.0)
        delta = path[1:] - path[:-1]
        jerk_scale = (60.0 - 360.0 * tau + 360.0 * tau**2) / durations[segment_indices] ** 3
        jerk = jerk_scale[:, None] * delta[segment_indices]
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

    def simulate_tracking(self, trajectory, *, gains=(4.0, 0.4), dt=None, damping=0.0, state=None):
        """Track a retimed or sampled joint trajectory in the virtual controller."""

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
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        if reference_time is not None and reference_time.size > 1:
            simulation_time = np.arange(reference_time[0], reference_time[-1], dt)
            if simulation_time.size == 0 or not np.isclose(simulation_time[-1], reference_time[-1]):
                simulation_time = np.append(simulation_time, reference_time[-1])
            reference = np.column_stack(
                [np.interp(simulation_time, reference_time, reference[:, joint]) for joint in range(self.n_joints)]
            )
        else:
            simulation_time = np.arange(reference.shape[0], dtype=float) * dt

        states, poses, controls = self.world.trajectory_follow(
            reference,
            gains=gains,
            dt=dt,
            damping=damping,
            state=state,
        )
        actual = states[1:, : self.n_joints]
        joint_error = actual - reference
        return RobotResult({
            "time": simulation_time,
            "reference": reference,
            "states": states,
            "poses": poses,
            "controls": controls,
            "actual": actual,
            "joint_error": joint_error,
            "dt": dt,
        })

    def safety_report(self, trajectory, tracking=None, *, clearance_margin=0.0, singularity_threshold=1e-8):
        """Summarize collision clearance, motion limits, and tracking error."""

        if isinstance(trajectory, Mapping):
            position = np.asarray(trajectory["position"], dtype=float)
            time = np.asarray(trajectory.get("time"), dtype=float)
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

        reference_clearances = np.asarray([self.world.clearance(q) for q in position], dtype=float)
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
            tracking_clearances = np.asarray([self.world.clearance(q) for q in actual], dtype=float)
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
            maximum = np.max(np.abs(values), axis=0)
            violation = None if limits is None else bool(np.any(maximum > np.asarray(limits) + 1e-12))
            motion_metrics[name] = {
                "maximum_by_joint": maximum.tolist(),
                "limits": None if limits is None else np.asarray(limits, dtype=float).tolist(),
                "violation": violation,
            }
        kinematics = [
            self.robot.kinematic_metrics(q, translational=True, threshold=singularity_threshold)
            for q in position
        ]
        condition_numbers = np.asarray([item["condition_number"] for item in kinematics], dtype=float)
        manipulabilities = np.asarray([item["manipulability"] for item in kinematics], dtype=float)
        clearance_violation = bool(np.any(all_clearances < clearance_margin))
        motion_violation = any(
            item is not None and item["violation"] is True for item in motion_metrics.values()
        )
        report = RobotResult({
            "workcell": self.name,
            "n_joints": self.n_joints,
            "n_samples": int(position.shape[0]),
            "duration": None if time is None else float(time[-1] - time[0]),
            "joint_path_length": joint_path_length,
            "collision": bool(np.any(all_clearances <= 0.0)),
            "clearance_margin": clearance_margin,
            "clearance_violation": clearance_violation,
            "reference_collision": bool(np.any(reference_clearances <= 0.0)),
            "tracking_collision": None if tracking_clearances is None else bool(np.any(tracking_clearances <= 0.0)),
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
            "max_velocity": None if velocity is None else float(np.max(np.abs(velocity))),
            "max_acceleration": None if acceleration is None else float(np.max(np.abs(acceleration))),
            "max_jerk": None if jerk is None else float(np.max(np.abs(jerk))),
            "motion_limits": motion_metrics,
            "motion_limit_violation": motion_violation,
            "minimum_position_manipulability": float(np.min(manipulabilities)),
            "maximum_position_condition_number": float(np.max(condition_numbers)),
            "singular_configuration": bool(any(item["singular"] for item in kinematics)),
        })
        if tracking is not None:
            error = np.asarray(tracking["joint_error"], dtype=float)
            if error.ndim != 2 or error.shape[1] != self.n_joints:
                raise ValueError("tracking joint_error must have shape (n_steps, n_joints)")
            error_norm = np.linalg.norm(error, axis=1)
            report["tracking_max_error"] = float(np.max(error_norm))
            report["tracking_rms_error"] = float(np.sqrt(np.mean(error_norm**2)))
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
