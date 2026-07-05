# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Cython helpers for trajectory generation hot paths."""

import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t


cdef inline void _time_scaling(double duration, double t, bint cubic, double* s, double* ds, double* dds):
    cdef double tau
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    tau = t / duration
    if tau < 0.0:
        tau = 0.0
    elif tau > 1.0:
        tau = 1.0
    if cubic:
        s[0] = 3.0 * tau * tau - 2.0 * tau * tau * tau
        ds[0] = (6.0 * tau - 6.0 * tau * tau) / duration
        dds[0] = (6.0 - 12.0 * tau) / (duration * duration)
    else:
        s[0] = 10.0 * tau * tau * tau - 15.0 * tau * tau * tau * tau + 6.0 * tau * tau * tau * tau * tau
        ds[0] = (30.0 * tau * tau - 60.0 * tau * tau * tau + 30.0 * tau * tau * tau * tau) / duration
        dds[0] = (60.0 * tau - 180.0 * tau * tau + 120.0 * tau * tau * tau) / (duration * duration)


def sample_joint_trajectory(object q0, object qf, double duration, int num_samples=100, kind="quintic"):
    """Sample a joint trajectory at evenly spaced times."""

    cdef np.ndarray[DTYPE_t, ndim=1] q0_arr = np.asarray(q0, dtype=np.float64).reshape(-1)
    cdef np.ndarray[DTYPE_t, ndim=1] qf_arr = np.asarray(qf, dtype=np.float64).reshape(-1)
    cdef np.ndarray[DTYPE_t, ndim=1] times = np.linspace(0.0, duration, num_samples)
    cdef np.ndarray[DTYPE_t, ndim=2] positions = np.empty((num_samples, q0_arr.shape[0]), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] velocities = np.empty_like(positions)
    cdef np.ndarray[DTYPE_t, ndim=2] accelerations = np.empty_like(positions)
    cdef np.ndarray[DTYPE_t, ndim=1] delta = qf_arr - q0_arr
    cdef Py_ssize_t i, j, n = q0_arr.shape[0]
    cdef double s = 0.0, ds = 0.0, dds = 0.0
    cdef bint cubic

    if q0_arr.shape[0] != qf_arr.shape[0]:
        raise ValueError("q0 and qf must have the same shape")
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if kind == "cubic":
        cubic = True
    elif kind == "quintic":
        cubic = False
    else:
        raise ValueError("kind must be 'cubic' or 'quintic'")

    for i in range(num_samples):
        _time_scaling(duration, times[i], cubic, &s, &ds, &dds)
        for j in range(n):
            positions[i, j] = q0_arr[j] + s * delta[j]
            velocities[i, j] = ds * delta[j]
            accelerations[i, j] = dds * delta[j]

    return times, positions, velocities, accelerations


def sample_joint_trajectory_segments(object q_waypoints, object durations, int num_samples_per_segment=100, kind="quintic"):
    """Sample a piecewise joint trajectory across multiple segments."""

    cdef np.ndarray[DTYPE_t, ndim=2] q = np.asarray(q_waypoints, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] ds = np.asarray(durations, dtype=np.float64).reshape(-1)
    cdef Py_ssize_t n_segments, n_joints, seg, i, j, total_samples = 0, seg_samples
    cdef bint cubic
    cdef double duration, offset = 0.0, s = 0.0, vel = 0.0, acc = 0.0
    cdef np.ndarray[DTYPE_t, ndim=1] t_out
    cdef np.ndarray[DTYPE_t, ndim=2] pos_out, vel_out, acc_out
    cdef np.ndarray[DTYPE_t, ndim=1] delta
    cdef np.ndarray[DTYPE_t, ndim=1] seg_times

    if q.ndim != 2:
        raise ValueError("q_waypoints must have shape (n_waypoints, n_joints)")
    if q.shape[0] < 2:
        raise ValueError("q_waypoints must contain at least two waypoints")
    if ds.shape[0] != q.shape[0] - 1:
        raise ValueError("durations must have one entry per segment")
    if num_samples_per_segment < 1:
        raise ValueError("num_samples_per_segment must be at least 1")

    if kind == "cubic":
        cubic = True
    elif kind == "quintic":
        cubic = False
    else:
        raise ValueError("kind must be 'cubic' or 'quintic'")

    n_segments = ds.shape[0]
    n_joints = q.shape[1]
    total_samples = num_samples_per_segment + (n_segments - 1) * (num_samples_per_segment - 1)
    t_out = np.empty(total_samples, dtype=np.float64)
    pos_out = np.empty((total_samples, n_joints), dtype=np.float64)
    vel_out = np.empty_like(pos_out)
    acc_out = np.empty_like(pos_out)

    seg_times = np.linspace(0.0, 1.0, num_samples_per_segment)
    seg = 0
    i = 0
    while seg < n_segments:
        duration = ds[seg]
        if duration <= 0.0:
            raise ValueError("durations must be positive")
        delta = q[seg + 1] - q[seg]
        for j in range(num_samples_per_segment):
            if seg > 0 and j == 0:
                continue
            _time_scaling(duration, seg_times[j] * duration, cubic, &s, &vel, &acc)
            t_out[i] = offset + seg_times[j] * duration
            pos_out[i, :] = q[seg] + s * delta
            vel_out[i, :] = vel * delta
            acc_out[i, :] = acc * delta
            i += 1
        offset += duration
        seg += 1

    return t_out, pos_out, vel_out, acc_out


def smooth_joint_path(object reference_waypoints, double smoothness=1.0, bint fixed_start=True, bint fixed_goal=True):
    """Smooth a joint-space path with a tridiagonal solver."""

    cdef np.ndarray[DTYPE_t, ndim=2] ref = np.asarray(reference_waypoints, dtype=np.float64)
    cdef Py_ssize_t n_waypoints, n_joints, start_idx, end_idx, m
    cdef np.ndarray[DTYPE_t, ndim=2] smoothed
    cdef np.ndarray[DTYPE_t, ndim=2] rhs
    cdef np.ndarray[DTYPE_t, ndim=1] main, upper, lower, cprime
    cdef Py_ssize_t i, j
    cdef double factor, diag_val, denom

    if ref.ndim != 2:
        raise ValueError("reference_waypoints must have shape (n_waypoints, n_joints)")
    n_waypoints = ref.shape[0]
    n_joints = ref.shape[1]
    if n_waypoints < 2:
        raise ValueError("reference_waypoints must contain at least two waypoints")
    if smoothness < 0.0:
        raise ValueError("smoothness must be non-negative")

    smoothed = ref.copy()
    if n_waypoints == 2 or smoothness == 0.0:
        if fixed_start:
            smoothed[0, :] = ref[0, :]
        if fixed_goal:
            smoothed[n_waypoints - 1, :] = ref[n_waypoints - 1, :]
        return smoothed

    start_idx = 1 if fixed_start else 0
    end_idx = n_waypoints - 1 if fixed_goal else n_waypoints
    m = end_idx - start_idx
    if m <= 0:
        return smoothed

    rhs = ref[start_idx:end_idx, :].copy()
    if fixed_start:
        rhs[0, :] += smoothness * ref[0, :]
    if fixed_goal:
        rhs[m - 1, :] += smoothness * ref[n_waypoints - 1, :]

    main = np.empty(m, dtype=np.float64)
    upper = np.empty(m - 1 if m > 1 else 1, dtype=np.float64)
    lower = np.empty(m - 1 if m > 1 else 1, dtype=np.float64)
    for i in range(m):
        if m == 1:
            diag_val = 2.0
        elif i == 0 or i == m - 1:
            diag_val = 1.0
        else:
            diag_val = 2.0
        main[i] = 1.0 + smoothness * diag_val
    if m > 1:
        upper[:m - 1] = -smoothness
        lower[:m - 1] = -smoothness
        cprime = np.empty(m - 1, dtype=np.float64)
        cprime[0] = upper[0] / main[0]
        for j in range(n_joints):
            rhs[0, j] = rhs[0, j] / main[0]
        for i in range(1, m - 1):
            factor = lower[i - 1]
            denom = main[i] - factor * cprime[i - 1]
            cprime[i] = upper[i] / denom
            for j in range(n_joints):
                rhs[i, j] = (rhs[i, j] - factor * rhs[i - 1, j]) / denom
        factor = lower[m - 2]
        denom = main[m - 1] - factor * cprime[m - 2]
        for j in range(n_joints):
            rhs[m - 1, j] = (rhs[m - 1, j] - factor * rhs[m - 2, j]) / denom
        for i in range(m - 2, -1, -1):
            for j in range(n_joints):
                rhs[i, j] = rhs[i, j] - cprime[i] * rhs[i + 1, j]
    else:
        for j in range(n_joints):
            rhs[0, j] = rhs[0, j] / main[0]

    smoothed[start_idx:end_idx, :] = rhs
    if fixed_start:
        smoothed[0, :] = ref[0, :]
    if fixed_goal:
        smoothed[n_waypoints - 1, :] = ref[n_waypoints - 1, :]
    return smoothed
