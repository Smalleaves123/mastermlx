from __future__ import annotations

import numpy as np

try:
    from ._trajectory_ops import sample_joint_trajectory as _cy_sample_joint_trajectory
    from ._trajectory_ops import sample_joint_trajectory_segments as _cy_sample_joint_trajectory_segments
except ImportError:  # pragma: no cover - fallback when Cython extensions are unavailable
    _cy_sample_joint_trajectory = None
    _cy_sample_joint_trajectory_segments = None


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

    if _cy_sample_joint_trajectory is not None:
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


def sample_joint_trajectory_segments(q_waypoints, durations, num_samples_per_segment=100, kind="quintic"):
    """Sample a piecewise joint trajectory across multiple segments."""

    if _cy_sample_joint_trajectory_segments is not None:
        return _cy_sample_joint_trajectory_segments(
            q_waypoints,
            durations,
            int(num_samples_per_segment),
            kind=kind,
        )

    q_waypoints = np.asarray(q_waypoints, dtype=float)
    durations = np.asarray(durations, dtype=float).reshape(-1)
    if q_waypoints.ndim != 2:
        raise ValueError("q_waypoints must have shape (n_waypoints, n_joints)")
    if q_waypoints.shape[0] < 2:
        raise ValueError("q_waypoints must contain at least two waypoints")
    if durations.size != q_waypoints.shape[0] - 1:
        raise ValueError("durations must have one entry per segment")

    times = []
    positions = []
    velocities = []
    accelerations = []
    offset = 0.0
    for idx in range(durations.size):
        seg_times, seg_pos, seg_vel, seg_acc = sample_joint_trajectory(
            q_waypoints[idx],
            q_waypoints[idx + 1],
            durations[idx],
            num_samples=num_samples_per_segment,
            kind=kind,
        )
        if idx > 0:
            seg_times = seg_times[1:]
            seg_pos = seg_pos[1:]
            seg_vel = seg_vel[1:]
            seg_acc = seg_acc[1:]
        times.append(seg_times + offset)
        positions.append(seg_pos)
        velocities.append(seg_vel)
        accelerations.append(seg_acc)
        offset += float(durations[idx])
    return (
        np.concatenate(times, axis=0),
        np.concatenate(positions, axis=0),
        np.concatenate(velocities, axis=0),
        np.concatenate(accelerations, axis=0),
    )


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
