# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython accelerated pairwise distance routines."""

import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t ITYPE_t


def pairwise_squared_euclidean(np.ndarray[DTYPE_t, ndim=2] X,
                                np.ndarray[DTYPE_t, ndim=2] Y):
    cdef ITYPE_t n = X.shape[0]
    cdef ITYPE_t m = Y.shape[0]
    cdef ITYPE_t d = X.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] out = np.empty((n, m), dtype=np.float64)
    cdef ITYPE_t i, j, k
    cdef DTYPE_t diff, acc

    for i in range(n):
        for j in range(m):
            acc = 0.0
            for k in range(d):
                diff = X[i, k] - Y[j, k]
                acc += diff * diff
            out[i, j] = acc
    return out


def pairwise_distances(np.ndarray[DTYPE_t, ndim=2] X,
                        np.ndarray[DTYPE_t, ndim=2] Y):
    cdef ITYPE_t n = X.shape[0]
    cdef ITYPE_t m = Y.shape[0]
    cdef ITYPE_t d = X.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] out = np.empty((n, m), dtype=np.float64)
    cdef ITYPE_t i, j, k
    cdef DTYPE_t diff, acc

    for i in range(n):
        for j in range(m):
            acc = 0.0
            for k in range(d):
                diff = X[i, k] - Y[j, k]
                acc += diff * diff
            out[i, j] = acc ** 0.5
    return out


def pairwise_manhattan_distances(np.ndarray[DTYPE_t, ndim=2] X,
                                   np.ndarray[DTYPE_t, ndim=2] Y):
    cdef ITYPE_t n = X.shape[0]
    cdef ITYPE_t m = Y.shape[0]
    cdef ITYPE_t d = X.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] out = np.empty((n, m), dtype=np.float64)
    cdef ITYPE_t i, j, k
    cdef DTYPE_t acc

    for i in range(n):
        for j in range(m):
            acc = 0.0
            for k in range(d):
                acc += abs(X[i, k] - Y[j, k])
            out[i, j] = acc
    return out
