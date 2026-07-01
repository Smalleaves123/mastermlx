# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Cython helpers for particle filter hot paths."""

import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t ITYPE_t


def systematic_resample(np.ndarray[DTYPE_t, ndim=1] weights, object rng=None):
    cdef Py_ssize_t n = weights.shape[0]
    cdef Py_ssize_t i, j
    cdef np.ndarray[DTYPE_t, ndim=1] norm = np.asarray(weights, dtype=np.float64).reshape(-1)
    cdef np.ndarray[ITYPE_t, ndim=1] indices = np.empty(n, dtype=np.intp)
    cdef np.ndarray[DTYPE_t, ndim=1] cumulative = np.empty(n, dtype=np.float64)
    cdef DTYPE_t total = 0.0, offset, step, threshold

    if n == 0:
        raise ValueError("weights cannot be empty")
    for i in range(n):
        total += norm[i]
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    for i in range(n):
        norm[i] /= total

    if rng is None:
        rng = np.random.default_rng()
    offset = float(rng.random())
    step = 1.0 / n
    cumulative[0] = norm[0]
    for i in range(1, n):
        cumulative[i] = cumulative[i - 1] + norm[i]

    j = 0
    for i in range(n):
        threshold = (offset + i) * step
        while j < n - 1 and cumulative[j] < threshold:
            j += 1
        indices[i] = j
    return indices


def normalize_weights(np.ndarray[DTYPE_t, ndim=1] weights):
    cdef Py_ssize_t n = weights.shape[0]
    cdef Py_ssize_t i
    cdef np.ndarray[DTYPE_t, ndim=1] out = np.asarray(weights, dtype=np.float64).reshape(-1).copy()
    cdef DTYPE_t total = 0.0

    if n == 0:
        raise ValueError("weights cannot be empty")
    for i in range(n):
        if out[i] < 0.0:
            out[i] = 0.0
        total += out[i]
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    for i in range(n):
        out[i] /= total
    return out
