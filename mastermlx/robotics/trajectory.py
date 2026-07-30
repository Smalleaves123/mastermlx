from __future__ import annotations

from functools import lru_cache
import importlib

import numpy as np

from ..config import get_backend

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
