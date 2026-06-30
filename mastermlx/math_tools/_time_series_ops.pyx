# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Cython helpers for time-series hot paths."""

import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t


def rolling_mean_1d(np.ndarray[DTYPE_t, ndim=1] x, int window):
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t out_n, i
    cdef double running
    cdef np.ndarray[DTYPE_t, ndim=1] out

    if n == 0:
        raise ValueError("x must be a non-empty 1D array")
    if window < 1:
        raise ValueError("window must be at least 1")
    if window > n:
        raise ValueError("window cannot exceed the series length")

    out_n = n - window + 1
    out = np.empty(out_n, dtype=np.float64)
    running = 0.0
    for i in range(window):
        running += x[i]
    out[0] = running / window
    for i in range(1, out_n):
        running += x[i + window - 1] - x[i - 1]
        out[i] = running / window
    return out


def autocorrelation_1d(np.ndarray[DTYPE_t, ndim=1] x, int lag, bint demean=True):
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t i
    cdef double mean = 0.0
    cdef double denom = 0.0
    cdef double num = 0.0
    cdef np.ndarray[DTYPE_t, ndim=1] centered

    if n == 0:
        raise ValueError("x must be a non-empty 1D array")
    if lag < 0:
        raise ValueError("lag must be non-negative")
    if lag == 0:
        return 1.0
    if lag >= n:
        return 0.0

    centered = np.asarray(x, dtype=np.float64).copy().reshape(-1)
    if demean:
        mean = 0.0
        for i in range(n):
            mean += centered[i]
        mean /= n
        for i in range(n):
            centered[i] -= mean

    for i in range(n):
        denom += centered[i] * centered[i]
    if denom == 0.0:
        return 0.0
    for i in range(n - lag):
        num += centered[i] * centered[i + lag]
    return num / denom


def autocorrelation_function_1d(np.ndarray[DTYPE_t, ndim=1] x, int max_lag, bint demean=True):
    cdef Py_ssize_t i
    cdef np.ndarray[DTYPE_t, ndim=1] out

    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    out = np.empty(max_lag + 1, dtype=np.float64)
    for i in range(max_lag + 1):
        out[i] = autocorrelation_1d(x, i, demean=demean)
    return out


def exponential_smoothing_1d(np.ndarray[DTYPE_t, ndim=1] x, double alpha):
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t i
    cdef np.ndarray[DTYPE_t, ndim=1] out

    if n == 0:
        raise ValueError("x must be a non-empty 1D array")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    out = np.empty(n, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, n):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def cusum_change_points_1d(np.ndarray[DTYPE_t, ndim=1] x, double threshold, double drift, int direction_code):
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t i
    cdef double mean = 0.0
    cdef double pos = 0.0
    cdef double neg = 0.0
    cdef double dev
    cdef list cps = []

    if n == 0:
        raise ValueError("x must be a non-empty 1D array")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")

    for i in range(n):
        mean += x[i]
    mean /= n

    for i in range(n):
        dev = x[i] - mean - drift
        pos = pos + dev
        if pos < 0.0:
            pos = 0.0
        neg = neg + dev
        if neg > 0.0:
            neg = 0.0
        if direction_code == 1:
            if pos > threshold:
                cps.append(i)
                pos = 0.0
                neg = 0.0
        elif direction_code == 2:
            if -neg > threshold:
                cps.append(i)
                pos = 0.0
                neg = 0.0
        else:
            if pos > threshold or -neg > threshold:
                cps.append(i)
                pos = 0.0
                neg = 0.0
    return cps
