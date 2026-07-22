"""Offline programming workflow for serial manipulators in a planar workcell."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..planning import smooth
from .model import RobotModel
from .trajectory import sample_joint_trajectory_segments


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


class RobotWorkcell:
    """Compose kinematics, collision planning, retiming, and virtual tracking.

    The workcell uses :class:`~mastermlx.sim.SimpleWorld`, so obstacle checks
    model planar circular obstacles.  It deliberately reuses the project's
    robot model and simulator instead of maintaining a second implementation.
    """

    def __init__(self, robot, world=None, name=None):
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

    @property
    def n_joints(self):
        return len(self.robot.links)

    def _collision_free_path(self, path, collision_step=0.05):
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[0] < 1 or path.shape[1] != self.n_joints:
            raise ValueError("path must have shape (n_points, n_joints)")
        collision_step = float(collision_step)
        if collision_step <= 0.0:
            raise ValueError("collision_step must be positive")
        for start, end in zip(path[:-1], path[1:]):
            count = max(1, int(np.ceil(np.linalg.norm(end - start) / collision_step)))
            for alpha in np.linspace(0.0, 1.0, count + 1):
                if self.world.hit(start + alpha * (end - start)):
                    return False
        return not self.world.hit(path[-1])

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

        return {
            "targets": normalized_targets,
            "joint_targets": np.asarray(configurations, dtype=float),
            "position_errors": np.asarray(position_errors, dtype=float),
            "orientation_errors": np.asarray(orientation_errors, dtype=float),
        }

    def plan_joint_path(
        self,
        q_start,
        q_goal,
        bounds,
        *,
        smooth_path=True,
        shortcut_attempts=100,
        collision_step=0.05,
        **rrt_kwargs,
    ):
        """Return a collision-free joint-space path, using a direct path first."""

        q_start = _joint_vector(q_start, self.n_joints, "q_start")
        q_goal = _joint_vector(q_goal, self.n_joints, "q_goal")
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (self.n_joints, 2) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("bounds must have shape (n_joints, 2) with lower < upper")
        if np.any(q_start < bounds[:, 0]) or np.any(q_start > bounds[:, 1]):
            raise ValueError("q_start must be inside bounds")
        if np.any(q_goal < bounds[:, 0]) or np.any(q_goal > bounds[:, 1]):
            raise ValueError("q_goal must be inside bounds")

        direct = np.vstack([q_start, q_goal])
        if self._collision_free_path(direct, collision_step=collision_step):
            return direct

        path = self.world.plan_path(q_start, q_goal, bounds, **rrt_kwargs)
        if path is None:
            raise RuntimeError("RRT could not find a collision-free joint-space path")
        if smooth_path:
            candidate = smooth(
                path,
                hit=self.world.hit,
                n=int(shortcut_attempts),
                random_state=rrt_kwargs.get("random_state"),
            )
            if self._collision_free_path(candidate, collision_step=collision_step):
                path = candidate
        if not self._collision_free_path(path, collision_step=collision_step):
            raise RuntimeError("planner returned a path that does not satisfy collision checks")
        return path

    def plan_tcp_task(self, targets, q_start, bounds, *, ik_kwargs=None, **planning_kwargs):
        """Plan a complete TCP task from continuous IK through collision-free motion."""

        ik_result = self.solve_tcp_path(targets, q_start, ik_kwargs=ik_kwargs)
        current = _joint_vector(q_start, self.n_joints, "q_start")
        segments: list[np.ndarray] = []
        for goal in ik_result["joint_targets"]:
            segment = self.plan_joint_path(current, goal, bounds, **planning_kwargs)
            segments.append(segment if not segments else segment[1:])
            current = goal
        path = np.concatenate(segments, axis=0)
        return {"ik": ik_result, "joint_path": path}

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
        return {
            "time": time,
            "position": position,
            "velocity": velocity,
            "acceleration": acceleration,
            "jerk": jerk,
            "durations": durations,
            "path": path.copy(),
        }

    def simulate_tracking(self, trajectory, *, gains=(4.0, 0.4), dt=None, damping=0.0, state=None):
        """Track a retimed or sampled joint trajectory in the virtual controller."""

        if isinstance(trajectory, dict):
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
        return {
            "time": simulation_time,
            "reference": reference,
            "states": states,
            "poses": poses,
            "controls": controls,
            "actual": actual,
            "joint_error": joint_error,
            "dt": dt,
        }

    def safety_report(self, trajectory, tracking=None):
        """Summarize collision clearance, motion limits, and tracking error."""

        if isinstance(trajectory, dict):
            position = np.asarray(trajectory["position"], dtype=float)
            time = np.asarray(trajectory.get("time"), dtype=float)
            velocity = trajectory.get("velocity")
            acceleration = trajectory.get("acceleration")
            jerk = trajectory.get("jerk")
        else:
            position = np.asarray(trajectory, dtype=float)
            time = None
            velocity = acceleration = jerk = None
        if position.ndim != 2 or position.shape[0] < 1 or position.shape[1] != self.n_joints:
            raise ValueError("trajectory positions must have shape (n_steps, n_joints)")

        clearances = np.asarray([self.world.clearance(q) for q in position], dtype=float)
        deltas = np.diff(position, axis=0)
        joint_path_length = float(np.sum(np.linalg.norm(deltas, axis=1))) if deltas.size else 0.0
        finite_clearances = clearances[np.isfinite(clearances)]
        report = {
            "workcell": self.name,
            "n_joints": self.n_joints,
            "n_samples": int(position.shape[0]),
            "duration": None if time is None else float(time[-1] - time[0]),
            "joint_path_length": joint_path_length,
            "collision": bool(np.any(clearances <= 0.0)),
            "minimum_clearance": None if finite_clearances.size == 0 else float(np.min(finite_clearances)),
            "max_velocity": None if velocity is None else float(np.max(np.abs(velocity))),
            "max_acceleration": None if acceleration is None else float(np.max(np.abs(acceleration))),
            "max_jerk": None if jerk is None else float(np.max(np.abs(jerk))),
        }
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

        if not isinstance(trajectory, dict):
            raise TypeError("trajectory must be the dictionary returned by retime_joint_path")
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

        paths = {"trajectory_csv": trajectory_path}
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
