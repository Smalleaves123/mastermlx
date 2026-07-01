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
