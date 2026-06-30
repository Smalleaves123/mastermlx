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
