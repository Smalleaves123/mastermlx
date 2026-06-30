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


def pairwise_cosine_distances(np.ndarray[DTYPE_t, ndim=2] X,
                               np.ndarray[DTYPE_t, ndim=2] Y):
    cdef ITYPE_t n = X.shape[0]
    cdef ITYPE_t m = Y.shape[0]
    cdef ITYPE_t d = X.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] out = np.empty((n, m), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] x_norms = np.empty(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] y_norms = np.empty(m, dtype=np.float64)
    cdef ITYPE_t i, j, k
    cdef DTYPE_t dot, xn2, yn2, denom

    for i in range(n):
        xn2 = 0.0
        for k in range(d):
            xn2 += X[i, k] * X[i, k]
        x_norms[i] = xn2 ** 0.5

    for j in range(m):
        yn2 = 0.0
        for k in range(d):
            yn2 += Y[j, k] * Y[j, k]
        y_norms[j] = yn2 ** 0.5

    for i in range(n):
        for j in range(m):
            dot = 0.0
            for k in range(d):
                dot += X[i, k] * Y[j, k]
            denom = x_norms[i] * y_norms[j]
            out[i, j] = 1.0 - dot / (denom if denom > 1e-12 else 1e-12)
    return out


def pairwise_hamming_distances(np.ndarray[DTYPE_t, ndim=2] X,
                                np.ndarray[DTYPE_t, ndim=2] Y):
    cdef ITYPE_t n = X.shape[0]
    cdef ITYPE_t m = Y.shape[0]
    cdef ITYPE_t d = X.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] out = np.empty((n, m), dtype=np.float64)
    cdef ITYPE_t i, j, k
    cdef DTYPE_t mismatches

    for i in range(n):
        for j in range(m):
            mismatches = 0.0
            for k in range(d):
                if X[i, k] != Y[j, k]:
                    mismatches += 1.0
            out[i, j] = mismatches / d
    return out


def pairwise_jaccard_distances(np.ndarray[DTYPE_t, ndim=2] X,
                                np.ndarray[DTYPE_t, ndim=2] Y):
    cdef ITYPE_t n = X.shape[0]
    cdef ITYPE_t m = Y.shape[0]
    cdef ITYPE_t d = X.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] out = np.empty((n, m), dtype=np.float64)
    cdef ITYPE_t i, j, k
    cdef DTYPE_t inter, uni
    cdef bint xk, yk

    for i in range(n):
        for j in range(m):
            inter = 0.0
            uni = 0.0
            for k in range(d):
                xk = X[i, k] != 0.0
                yk = Y[j, k] != 0.0
                if xk and yk:
                    inter += 1.0
                if xk or yk:
                    uni += 1.0
            out[i, j] = (uni - inter) / (uni if uni > 1e-12 else 1e-12)
    return out


def pairwise_mahalanobis_distances(np.ndarray[DTYPE_t, ndim=2] X,
                                    np.ndarray[DTYPE_t, ndim=2] Y,
                                    np.ndarray[DTYPE_t, ndim=2] VI):
    cdef ITYPE_t n = X.shape[0]
    cdef ITYPE_t m = Y.shape[0]
    cdef ITYPE_t d = X.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] out = np.empty((n, m), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] diff = np.empty(d, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] tmp = np.empty(d, dtype=np.float64)
    cdef ITYPE_t i, j, k, l
    cdef DTYPE_t acc, diff_k

    if VI.shape[0] != VI.shape[1] or VI.shape[0] != d:
        raise ValueError("VI must be a square matrix matching the number of features")

    for i in range(n):
        for j in range(m):
            for k in range(d):
                diff[k] = X[i, k] - Y[j, k]
            for k in range(d):
                acc = 0.0
                for l in range(d):
                    acc += VI[k, l] * diff[l]
                tmp[k] = acc
            acc = 0.0
            for k in range(d):
                diff_k = diff[k]
                acc += diff_k * tmp[k]
            out[i, j] = acc ** 0.5
    return out
