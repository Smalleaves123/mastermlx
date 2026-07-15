# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Cython kernels for IIR signal filtering and ridge extraction."""

import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t ITYPE_t

np.import_array()


def iir_filter_1d(np.ndarray[DTYPE_t, ndim=1] x,
                   np.ndarray[DTYPE_t, ndim=1] b,
                   np.ndarray[DTYPE_t, ndim=1] a):
    cdef ITYPE_t n = x.shape[0]
    cdef ITYPE_t nb = b.shape[0]
    cdef ITYPE_t na = a.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] y = np.zeros(n, dtype=np.float64)
    cdef ITYPE_t i, k, upper
    cdef double value

    for i in range(n):
        value = 0.0
        upper = nb if nb < i + 1 else i + 1
        for k in range(upper):
            value += b[k] * x[i - k]
        upper = na if na < i + 1 else i + 1
        for k in range(1, upper):
            value -= a[k] * y[i - k]
        y[i] = value
    return y


def ridge_path(np.ndarray[DTYPE_t, ndim=2] score, double smoothness, int max_jump=-1):
    cdef ITYPE_t n_freqs = score.shape[0]
    cdef ITYPE_t n_times = score.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] dynamic = np.full((n_freqs, n_times), -np.inf, dtype=np.float64)
    cdef np.ndarray[ITYPE_t, ndim=2] back = np.zeros((n_freqs, n_times), dtype=np.intp)
    cdef np.ndarray[ITYPE_t, ndim=1] indices = np.zeros(n_times, dtype=np.intp)
    cdef ITYPE_t t, current, previous, low, high, best
    cdef double candidate, best_value, value

    for current in range(n_freqs):
        dynamic[current, 0] = score[current, 0]
    for t in range(1, n_times):
        for current in range(n_freqs):
            low = 0 if max_jump < 0 else max(0, current - max_jump)
            high = n_freqs if max_jump < 0 else min(n_freqs, current + max_jump + 1)
            best = low
            best_value = -np.inf
            for previous in range(low, high):
                candidate = dynamic[previous, t - 1] - smoothness * (previous - current) * (previous - current)
                if candidate > best_value:
                    best_value = candidate
                    best = previous
            back[current, t] = best
            dynamic[current, t] = score[current, t] + best_value
    best = 0
    best_value = dynamic[0, n_times - 1]
    for current in range(1, n_freqs):
        value = dynamic[current, n_times - 1]
        if value > best_value:
            best_value = value
            best = current
    indices[n_times - 1] = best
    for t in range(n_times - 1, 0, -1):
        indices[t - 1] = back[indices[t], t]
    return indices
