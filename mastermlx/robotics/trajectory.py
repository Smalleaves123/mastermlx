from __future__ import annotations

from functools import lru_cache
import importlib

import numpy as np

from ..config import get_backend
from .constraints import validate_joint_limits
from .results import JointTrajectory, RobotResult

try:
    from ._trajectory_ops import sample_joint_trajectory as _cy_sample_joint_trajectory
    from ._trajectory_ops import sample_joint_trajectory_segments as _cy_sample_joint_trajectory_segments
    from ._trajectory_ops import smooth_joint_path as _cy_smooth_joint_path
except ImportError:  # pragma: no cover - fallback when Cython extensions are unavailable
    _cy_sample_joint_trajectory = None
    _cy_sample_joint_trajectory_segments = None
    _cy_smooth_joint_path = None


@lru_cache(maxsize=3)
def _load_cpp_retiming(backend=None):
    """Load the optional C++ quintic retiming kernel for the auto backend."""

    if backend is None:
        backend = get_backend()
    if backend != "auto":
        return None
    try:
        return importlib.import_module("mastermlx.robotics._retiming_cpp")
    except ImportError:
        return None


def _retime_quintic_path_compiled(
    path, velocity_limits, acceleration_limits, jerk_limits, num_samples_per_segment, minimum_duration
):
    """Return compiled retiming arrays when the optional C++ kernel is available."""

    cpp = _load_cpp_retiming(get_backend())
    if cpp is None or not callable(getattr(cpp, "retime_quintic_path", None)):
        return None
    return tuple(
        np.asarray(value, dtype=float)
        for value in cpp.retime_quintic_path(
            np.ascontiguousarray(path, dtype=float),
            np.ascontiguousarray(velocity_limits, dtype=float),
            None if acceleration_limits is None else np.ascontiguousarray(acceleration_limits, dtype=float),
            None if jerk_limits is None else np.ascontiguousarray(jerk_limits, dtype=float),
            int(num_samples_per_segment),
            float(minimum_duration),
        )
    )


def trajectory_peaks_batch(values, *, output=None):
    """Return maximum absolute trajectory values per metric and joint."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or min(values.shape) < 1:
        raise ValueError("values must have shape (n_samples, n_joints, n_metrics)")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must contain only finite values")
    shape = (values.shape[2], values.shape[1])
    if output is None:
        result = np.empty(shape, dtype=float)
    elif isinstance(output, np.ndarray) and output.dtype == np.dtype(float) and output.flags.c_contiguous:
        if output.shape != shape:
            raise ValueError(f"output must have shape {shape}")
        result = output
    else:
        raise ValueError("output must be a contiguous float64 NumPy array")

    cpp = _load_cpp_retiming(get_backend())
    if cpp is not None and callable(getattr(cpp, "trajectory_peaks_batch", None)):
        return np.asarray(
            cpp.trajectory_peaks_batch(np.ascontiguousarray(values), result), dtype=float
        )
    result[...] = np.max(np.abs(values), axis=0).T
    return result


def _normalize_time(t, duration):
    duration = float(duration)
    if duration <= 0:
        raise ValueError("duration must be positive")
    tau = np.asarray(t, dtype=float) / duration
    return np.clip(tau, 0.0, 1.0), duration


def cubic_time_scaling(duration, t):
    tau, duration = _normalize_time(t, duration)
    s = 3.0 * tau**2 - 2.0 * tau**3
    ds = (6.0 * tau - 6.0 * tau**2) / duration
    dds = (6.0 - 12.0 * tau) / (duration**2)
    return s, ds, dds


def quintic_time_scaling(duration, t):
    tau, duration = _normalize_time(t, duration)
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    ds = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / duration
    dds = (60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3) / (duration**2)
    return s, ds, dds


def joint_trajectory(q0, qf, duration, t, kind="quintic"):
    """Interpolate between joint configurations with smooth time scaling."""

    q0 = np.asarray(q0, dtype=float)
    qf = np.asarray(qf, dtype=float)
    if q0.shape != qf.shape:
        raise ValueError("q0 and qf must have the same shape")
    delta = qf - q0
    if kind == "cubic":
        s, ds, dds = cubic_time_scaling(duration, t)
    elif kind == "quintic":
        s, ds, dds = quintic_time_scaling(duration, t)
    else:
        raise ValueError("kind must be 'cubic' or 'quintic'")
    return q0 + s * delta, ds * delta, dds * delta


def sample_joint_trajectory(q0, qf, duration, num_samples=100, kind="quintic"):
    """Sample a joint trajectory at evenly spaced times."""

    if get_backend() != "numpy" and _cy_sample_joint_trajectory is not None:
        return _cy_sample_joint_trajectory(q0, qf, float(duration), int(num_samples), kind=kind)

    times = np.linspace(0.0, float(duration), int(num_samples))
    positions = []
    velocities = []
    accelerations = []
    for t in times:
        q, qd, qdd = joint_trajectory(q0, qf, duration, t, kind=kind)
        positions.append(q)
        velocities.append(qd)
        accelerations.append(qdd)
    return times, np.asarray(positions), np.asarray(velocities), np.asarray(accelerations)


def _trajectory_output_buffers(output, total_samples, n_joints):
    shapes = {
        "time": ((total_samples,), float),
        "position": ((total_samples, n_joints), float),
        "velocity": ((total_samples, n_joints), float),
        "acceleration": ((total_samples, n_joints), float),
    }
    if output is None:
        return {
            key: np.empty(shape, dtype=dtype) for key, (shape, dtype) in shapes.items()
        }
    if not isinstance(output, dict):
        raise ValueError("output must be a mapping of named contiguous NumPy arrays")
    buffers = {}
    for key, (shape, dtype) in shapes.items():
        value = output.get(key)
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype(dtype)
            or not value.flags.c_contiguous
            or value.shape != shape
        ):
            raise ValueError(f"output[{key!r}] must be a contiguous float64 array with shape {shape}")
        buffers[key] = value
    return buffers


def sample_joint_trajectory_segments(
    q_waypoints, durations, num_samples_per_segment=100, kind="quintic", *, output=None
):
    """Sample a piecewise joint trajectory across multiple segments."""

    q_waypoints = np.asarray(q_waypoints, dtype=float)
    durations = np.asarray(durations, dtype=float).reshape(-1)
    if q_waypoints.ndim != 2:
        raise ValueError("q_waypoints must have shape (n_waypoints, n_joints)")
    if q_waypoints.shape[0] < 2:
        raise ValueError("q_waypoints must contain at least two waypoints")
    if durations.size != q_waypoints.shape[0] - 1:
        raise ValueError("durations must have one entry per segment")
    if not np.all(np.isfinite(q_waypoints)) or not np.all(np.isfinite(durations)):
        raise ValueError("q_waypoints and durations must contain only finite values")
    if np.any(durations <= 0.0):
        raise ValueError("durations must be positive")
    num_samples_per_segment = int(num_samples_per_segment)
    if num_samples_per_segment < 1:
        raise ValueError("num_samples_per_segment must be at least 1")
    if kind not in {"cubic", "quintic"}:
        raise ValueError("kind must be 'cubic' or 'quintic'")

    segments = durations.size
    total_samples = num_samples_per_segment + (segments - 1) * (num_samples_per_segment - 1)
    buffers = _trajectory_output_buffers(output, total_samples, q_waypoints.shape[1])

    cpp = _load_cpp_retiming(get_backend())
    if cpp is not None and callable(getattr(cpp, "sample_joint_trajectory_segments", None)):
        values = cpp.sample_joint_trajectory_segments(
            np.ascontiguousarray(q_waypoints),
            np.ascontiguousarray(durations),
            num_samples_per_segment,
            kind,
            buffers["time"],
            buffers["position"],
            buffers["velocity"],
            buffers["acceleration"],
        )
        return tuple(values)

    if get_backend() != "numpy" and _cy_sample_joint_trajectory_segments is not None:
        values = _cy_sample_joint_trajectory_segments(
            q_waypoints, durations, num_samples_per_segment, kind=kind
        )
        for key, value in zip(buffers, values):
            buffers[key][...] = value
        return tuple(buffers.values())

    times = buffers["time"]
    positions = buffers["position"]
    velocities = buffers["velocity"]
    accelerations = buffers["acceleration"]
    output_index = 0
    offset = 0.0
    for idx in range(durations.size):
        duration = durations[idx]
        for sample in range(num_samples_per_segment):
            if idx > 0 and sample == 0:
                continue
            t = 0.0 if num_samples_per_segment == 1 else sample * duration / (num_samples_per_segment - 1)
            q, qd, qdd = joint_trajectory(
                q_waypoints[idx], q_waypoints[idx + 1], duration, t, kind=kind
            )
            times[output_index] = t + offset
            positions[output_index] = q
            velocities[output_index] = qd
            accelerations[output_index] = qdd
            output_index += 1
        offset += float(durations[idx])
    return times, positions, velocities, accelerations


def plan_joint_path(q_start, q_goal, num_waypoints=11, via_points=None):
    """Generate a piecewise-linear joint-space path."""

    q_start = np.asarray(q_start, dtype=float).reshape(-1)
    q_goal = np.asarray(q_goal, dtype=float).reshape(-1)
    if q_start.shape != q_goal.shape:
        raise ValueError("q_start and q_goal must have the same shape")
    if int(num_waypoints) < 2:
        raise ValueError("num_waypoints must be at least 2")

    if via_points is None:
        alphas = np.linspace(0.0, 1.0, int(num_waypoints))[:, None]
        return q_start[None, :] + alphas * (q_goal - q_start)[None, :]

    waypoints = [q_start]
    for point in via_points:
        point = np.asarray(point, dtype=float).reshape(-1)
        if point.shape != q_start.shape:
            raise ValueError("via_points must match the joint dimension")
        waypoints.append(point)
    waypoints.append(q_goal)
    return np.asarray(waypoints, dtype=float)


def smooth_joint_path(reference_waypoints, smoothness=1.0, fixed_start=True, fixed_goal=True):
    """Smooth a joint-space path with a quadratic first-difference penalty.

    The optimizer keeps the start and goal fixed by default and solves a
    tridiagonal least-squares system for the intermediate waypoints.
    """

    if get_backend() != "numpy" and _cy_smooth_joint_path is not None:
        return _cy_smooth_joint_path(
            reference_waypoints,
            float(smoothness),
            bool(fixed_start),
            bool(fixed_goal),
        )

    reference_waypoints = np.asarray(reference_waypoints, dtype=float)
    if reference_waypoints.ndim != 2:
        raise ValueError("reference_waypoints must have shape (n_waypoints, n_joints)")
    n_waypoints, n_joints = reference_waypoints.shape
    if n_waypoints < 2:
        raise ValueError("reference_waypoints must contain at least two waypoints")

    smoothness = float(smoothness)
    if smoothness < 0.0:
        raise ValueError("smoothness must be non-negative")

    if n_waypoints == 2 or smoothness == 0.0:
        smoothed = reference_waypoints.copy()
        if fixed_start:
            smoothed[0] = reference_waypoints[0]
        if fixed_goal:
            smoothed[-1] = reference_waypoints[-1]
        return smoothed

    smoothed = reference_waypoints.copy()
    indices = np.arange(n_waypoints)
    if fixed_start:
        indices = indices[1:]
    if fixed_goal:
        indices = indices[:-1]

    if indices.size == 0:
        return smoothed

    interior = indices
    m = interior.size
    A = np.eye(m, dtype=float)
    if m > 1:
        diag = np.ones(m, dtype=float) * 2.0
        diag[0] = 1.0
        diag[-1] = 1.0
        A += smoothness * np.diag(diag)
        off = -smoothness * np.ones(m - 1, dtype=float)
        A += np.diag(off, k=1) + np.diag(off, k=-1)
    else:
        A += np.array([[2.0 * smoothness]], dtype=float)

    for j in range(n_joints):
        b = reference_waypoints[interior, j].copy()
        if fixed_start:
            b[0] += smoothness * reference_waypoints[0, j]
        if fixed_goal:
            b[-1] += smoothness * reference_waypoints[-1, j]
        smoothed[interior, j] = np.linalg.solve(A, b)

    if fixed_start:
        smoothed[0] = reference_waypoints[0]
    if fixed_goal:
        smoothed[-1] = reference_waypoints[-1]
    return smoothed


_QUINTIC_MAX_VELOCITY = 1.875
_QUINTIC_MAX_ACCELERATION = 10.0 / np.sqrt(3.0)
_QUINTIC_MAX_JERK = 60.0


def _constraint_durations(path, velocity_limits, acceleration_limits, jerk_limits, minimum_duration):
    durations = []
    for delta in np.abs(np.diff(path, axis=0)):
        candidates = []
        if velocity_limits is not None:
            candidates.append(_QUINTIC_MAX_VELOCITY * delta / velocity_limits)
        if acceleration_limits is not None:
            candidates.append(np.sqrt(_QUINTIC_MAX_ACCELERATION * delta / acceleration_limits))
        if jerk_limits is not None:
            candidates.append(np.cbrt(_QUINTIC_MAX_JERK * delta / jerk_limits))
        durations.append(max(float(minimum_duration), float(np.max(np.concatenate(candidates)))))
    return np.asarray(durations, dtype=float)


def _retime_path_with_limits(
    path,
    durations,
    *,
    num_samples_per_segment,
    velocity_limits,
    acceleration_limits,
    jerk_limits,
):
    time, position, velocity, acceleration = sample_joint_trajectory_segments(
        path,
        durations,
        num_samples_per_segment=num_samples_per_segment,
        kind="quintic",
    )
    jerk_parts = []
    for segment, duration in enumerate(durations):
        tau = np.linspace(0.0, 1.0, num_samples_per_segment)
        if segment > 0:
            tau = tau[1:]
        jerk_scale = (60.0 - 360.0 * tau + 360.0 * tau**2) / duration**3
        jerk_parts.append(jerk_scale[:, None] * (path[segment + 1] - path[segment]))
    jerk = np.concatenate(jerk_parts, axis=0)
    return JointTrajectory(
        time=time,
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        jerk=jerk,
        durations=durations,
        path=path.copy(),
        velocity_limits=None if velocity_limits is None else velocity_limits.copy(),
        acceleration_limits=None if acceleration_limits is None else acceleration_limits.copy(),
        jerk_limits=None if jerk_limits is None else jerk_limits.copy(),
    )


def optimize_joint_path(
    reference_waypoints,
    *,
    smoothness=1.0,
    reference_weight=1.0,
    path_cost=None,
    path_cost_weight=1.0,
    joint_limits=None,
    max_iter=100,
    step_size=0.1,
    tolerance=1e-6,
    finite_difference_eps=1e-5,
    fixed_start=True,
    fixed_goal=True,
    velocity_limits=None,
    acceleration_limits=None,
    jerk_limits=None,
    segment_durations=None,
    num_samples_per_segment=101,
    minimum_duration=1e-3,
):
    """Optimize a joint-space path with smoothness and optional path cost.

    The objective combines a reference-path penalty with a squared
    second-difference (curvature) penalty. ``path_cost`` may add a scalar cost
    for the complete path, for example a collision or clearance barrier. Its
    gradient is estimated only for movable interior waypoints, while the
    smoothness gradient is analytic. Each update is projected into
    ``joint_limits`` and accepted only when it reduces the objective.
    """

    reference = np.asarray(reference_waypoints, dtype=float)
    if reference.ndim != 2 or reference.shape[0] < 2 or reference.shape[1] < 1:
        raise ValueError("reference_waypoints must have shape (n_waypoints, n_joints)")
    if not np.all(np.isfinite(reference)):
        raise ValueError("reference_waypoints must contain only finite values")

    smoothness = float(smoothness)
    reference_weight = float(reference_weight)
    path_cost_weight = float(path_cost_weight)
    step_size = float(step_size)
    tolerance = float(tolerance)
    finite_difference_eps = float(finite_difference_eps)
    max_iter = int(max_iter)
    if smoothness < 0.0 or reference_weight < 0.0 or path_cost_weight < 0.0:
        raise ValueError("cost weights must be non-negative")
    if step_size <= 0.0 or tolerance <= 0.0 or finite_difference_eps <= 0.0:
        raise ValueError("step_size, tolerance, and finite_difference_eps must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if path_cost is not None and not callable(path_cost):
        raise TypeError("path_cost must be callable or None")

    limits = validate_joint_limits(joint_limits, reference.shape[1])
    motion_limits = [velocity_limits, acceleration_limits, jerk_limits]
    normalized_motion_limits: list[np.ndarray | None] | None = None
    if any(value is not None for value in motion_limits):
        if velocity_limits is None and acceleration_limits is None and jerk_limits is None:
            raise ValueError("at least one motion limit must be provided")
        normalized_motion_limits = []
        for value, name in zip(
            motion_limits, ("velocity_limits", "acceleration_limits", "jerk_limits")
        ):
            if value is None:
                normalized_motion_limits.append(None)
                continue
            array = np.asarray(value, dtype=float)
            if array.ndim == 0:
                array = np.full(reference.shape[1], float(array))
            else:
                array = array.reshape(-1)
            if (
                array.shape != (reference.shape[1],)
                or not np.all(np.isfinite(array))
                or np.any(array <= 0.0)
            ):
                raise ValueError(f"{name} must be a positive scalar or vector of length n_joints")
            normalized_motion_limits.append(array)
        velocity_limits, acceleration_limits, jerk_limits = normalized_motion_limits
        minimum_duration = float(minimum_duration)
        num_samples_per_segment = int(num_samples_per_segment)
        if minimum_duration <= 0.0 or not np.isfinite(minimum_duration):
            raise ValueError("minimum_duration must be positive and finite")
        if num_samples_per_segment < 2:
            raise ValueError("num_samples_per_segment must be at least 2")
        if segment_durations is not None:
            durations = np.asarray(segment_durations, dtype=float).reshape(-1)
            if durations.size != reference.shape[0] - 1 or not np.all(np.isfinite(durations)):
                raise ValueError("segment_durations must have one finite value per path segment")
            if np.any(durations < minimum_duration):
                raise ValueError("segment_durations must be at least minimum_duration")
        else:
            durations = _constraint_durations(
                reference,
                velocity_limits,
                acceleration_limits,
                jerk_limits,
                minimum_duration,
            )
    else:
        normalized_motion_limits = None
        durations = None
    path = reference.copy()
    if limits is not None and (
        np.any(path < limits[:, 0]) or np.any(path > limits[:, 1])
    ):
        raise ValueError("reference_waypoints exceed joint_limits")

    movable = np.arange(reference.shape[0])
    if fixed_start:
        movable = movable[1:]
    if fixed_goal:
        movable = movable[:-1]

    def _custom_cost(candidate):
        if path_cost is None or path_cost_weight == 0.0:
            return 0.0
        value = float(path_cost(candidate))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("path_cost must return a finite non-negative scalar")
        return path_cost_weight * value

    def _components(candidate):
        differences = candidate[2:] - 2.0 * candidate[1:-1] + candidate[:-2]
        reference_value = reference_weight * float(np.sum((candidate - reference) ** 2))
        smooth_value = smoothness * float(np.sum(differences**2))
        custom_value = _custom_cost(candidate)
        return reference_value, smooth_value, custom_value

    def _objective(candidate):
        return float(sum(_components(candidate)))

    def _analytic_gradient(candidate):
        gradient = 2.0 * reference_weight * (candidate - reference)
        if smoothness != 0.0 and candidate.shape[0] > 2:
            differences = candidate[2:] - 2.0 * candidate[1:-1] + candidate[:-2]
            gradient[:-2] += 2.0 * smoothness * differences
            gradient[1:-1] -= 4.0 * smoothness * differences
            gradient[2:] += 2.0 * smoothness * differences
        return gradient

    def _custom_gradient(candidate):
        gradient = np.zeros_like(candidate)
        if path_cost is None or path_cost_weight == 0.0:
            return gradient
        for row in movable:
            for column in range(candidate.shape[1]):
                plus = candidate.copy()
                minus = candidate.copy()
                plus[row, column] += finite_difference_eps
                minus[row, column] -= finite_difference_eps
                if limits is not None:
                    plus[row, column] = np.clip(
                        plus[row, column], limits[column, 0], limits[column, 1]
                    )
                    minus[row, column] = np.clip(
                        minus[row, column], limits[column, 0], limits[column, 1]
                    )
                gradient[row, column] = (
                    _custom_cost(plus) - _custom_cost(minus)
                ) / (2.0 * finite_difference_eps)
        return gradient

    initial_components = _components(path)
    initial_cost = float(sum(initial_components))
    history = [initial_cost]
    converged = not movable.size
    message = "no movable interior waypoints" if converged else "maximum iterations reached"
    iterations = 0

    for iterations in range(1, max_iter + 1):
        gradient = _analytic_gradient(path) + _custom_gradient(path)
        if fixed_start:
            gradient[0] = 0.0
        if fixed_goal:
            gradient[-1] = 0.0
        gradient_norm = float(np.linalg.norm(gradient[movable])) if movable.size else 0.0
        if gradient_norm <= tolerance:
            converged = True
            message = "gradient tolerance reached"
            break

        current_cost = history[-1]
        accepted = False
        local_step = step_size
        for _ in range(20):
            candidate = path.copy()
            candidate[movable] -= local_step * gradient[movable]
            if limits is not None:
                candidate = np.clip(candidate, limits[:, 0], limits[:, 1])
            if fixed_start:
                candidate[0] = reference[0]
            if fixed_goal:
                candidate[-1] = reference[-1]
            candidate_cost = _objective(candidate)
            if candidate_cost < current_cost - tolerance * max(1.0, abs(current_cost)):
                path = candidate
                history.append(candidate_cost)
                accepted = True
                break
            local_step *= 0.5
        if not accepted:
            converged = True
            message = "line search reached a stationary point"
            break

    final_components = _components(path)
    result = {
        "path": path,
        "initial_cost": initial_cost,
        "final_cost": float(sum(final_components)),
        "reference_cost": final_components[0],
        "smoothness_cost": final_components[1],
        "path_cost": final_components[2],
        "iterations": iterations,
        "converged": converged,
        "message": message,
        "history": np.asarray(history, dtype=float),
        "fixed_start": bool(fixed_start),
        "fixed_goal": bool(fixed_goal),
    }
    if normalized_motion_limits is not None:
        trajectory = _retime_path_with_limits(
            path,
            durations,
            num_samples_per_segment=num_samples_per_segment,
            velocity_limits=velocity_limits,
            acceleration_limits=acceleration_limits,
            jerk_limits=jerk_limits,
        )
        result["trajectory"] = trajectory
        result["motion_limits"] = {
            "velocity": None if velocity_limits is None else velocity_limits.copy(),
            "acceleration": None if acceleration_limits is None else acceleration_limits.copy(),
            "jerk": None if jerk_limits is None else jerk_limits.copy(),
        }
    return RobotResult(result)


def plan_joint_trajectory(q_start, q_goal, duration, num_waypoints=11, num_samples_per_segment=100, kind="quintic", smoothness=0.0, via_points=None):
    """Plan and sample a joint trajectory from a start and goal configuration."""

    path = plan_joint_path(q_start, q_goal, num_waypoints=num_waypoints, via_points=via_points)
    if smoothness > 0.0:
        path = smooth_joint_path(path, smoothness=smoothness)
    segments = path.shape[0] - 1
    if segments < 1:
        raise ValueError("at least two waypoints are required")
    durations = np.full(segments, float(duration) / segments, dtype=float)
    return sample_joint_trajectory_segments(path, durations, num_samples_per_segment=num_samples_per_segment, kind=kind)
