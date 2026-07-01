# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Cython helpers for scalar/vector distance hot paths."""

import numpy as np
cimport numpy as np
from libc.math cimport fabs, pow, sqrt

ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t ITYPE_t


def euclidean_distance(object a, object b):
    cdef np.ndarray[DTYPE_t, ndim=1] A = np.asarray(a, dtype=np.float64).reshape(-1)
    cdef np.ndarray[DTYPE_t, ndim=1] B = np.asarray(b, dtype=np.float64).reshape(-1)
    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t i
    cdef DTYPE_t acc = 0.0, diff
    if A.shape[0] != B.shape[0]:
        raise ValueError("a and b must have the same shape")
    for i in range(n):
        diff = A[i] - B[i]
        acc += diff * diff
    return sqrt(acc)


def manhattan_distance(object a, object b):
    cdef np.ndarray[DTYPE_t, ndim=1] A = np.asarray(a, dtype=np.float64).reshape(-1)
    cdef np.ndarray[DTYPE_t, ndim=1] B = np.asarray(b, dtype=np.float64).reshape(-1)
    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t i
    cdef DTYPE_t acc = 0.0
    if A.shape[0] != B.shape[0]:
        raise ValueError("a and b must have the same shape")
    for i in range(n):
        acc += fabs(A[i] - B[i])
    return acc


def minkowski_distance(object a, object b, double p=2.0):
    cdef np.ndarray[DTYPE_t, ndim=1] A = np.asarray(a, dtype=np.float64).reshape(-1)
    cdef np.ndarray[DTYPE_t, ndim=1] B = np.asarray(b, dtype=np.float64).reshape(-1)
    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t i
    cdef DTYPE_t acc = 0.0
    if p <= 0.0:
        raise ValueError("p must be positive")
    if A.shape[0] != B.shape[0]:
        raise ValueError("a and b must have the same shape")
    for i in range(n):
        acc += pow(fabs(A[i] - B[i]), p)
    return pow(acc, 1.0 / p)


def chebyshev_distance(object a, object b):
    cdef np.ndarray[DTYPE_t, ndim=1] A = np.asarray(a, dtype=np.float64).reshape(-1)
    cdef np.ndarray[DTYPE_t, ndim=1] B = np.asarray(b, dtype=np.float64).reshape(-1)
    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t i
    cdef DTYPE_t best = 0.0, diff
    if A.shape[0] != B.shape[0]:
        raise ValueError("a and b must have the same shape")
    for i in range(n):
        diff = fabs(A[i] - B[i])
        if diff > best:
            best = diff
    return best


def cosine_distance(object a, object b):
    cdef np.ndarray[DTYPE_t, ndim=1] A = np.asarray(a, dtype=np.float64).reshape(-1)
    cdef np.ndarray[DTYPE_t, ndim=1] B = np.asarray(b, dtype=np.float64).reshape(-1)
    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t i
    cdef DTYPE_t dot = 0.0, an2 = 0.0, bn2 = 0.0, denom
    if A.shape[0] != B.shape[0]:
        raise ValueError("a and b must have the same shape")
    for i in range(n):
        dot += A[i] * B[i]
        an2 += A[i] * A[i]
        bn2 += B[i] * B[i]
    denom = sqrt(an2) * sqrt(bn2)
    return 1.0 - dot / (denom if denom > 1e-12 else 1e-12)


def hamming_distance(object a, object b):
    cdef np.ndarray AA = np.asarray(a)
    cdef np.ndarray BB = np.asarray(b)
    cdef np.ndarray AA_flat = AA.reshape(-1)
    cdef np.ndarray BB_flat = BB.reshape(-1)
    cdef Py_ssize_t n = AA.size
    cdef Py_ssize_t i
    cdef DTYPE_t acc = 0.0
    if AA.size != BB.size:
        raise ValueError("a and b must have the same shape")
    for i in range(n):
        if AA_flat[i] != BB_flat[i]:
            acc += 1.0
    return acc / n


def jaccard_distance(object a, object b):
    cdef np.ndarray AA = np.asarray(a).astype(bool)
    cdef np.ndarray BB = np.asarray(b).astype(bool)
    cdef np.ndarray AA_flat = AA.reshape(-1)
    cdef np.ndarray BB_flat = BB.reshape(-1)
    cdef Py_ssize_t n = AA.size
    cdef Py_ssize_t i
    cdef DTYPE_t inter = 0.0, uni = 0.0
    if AA.size != BB.size:
        raise ValueError("a and b must have the same shape")
    for i in range(n):
        if AA_flat[i] and BB_flat[i]:
            inter += 1.0
        if AA_flat[i] or BB_flat[i]:
            uni += 1.0
    return (uni - inter) / (uni if uni > 1e-12 else 1e-12)


def mahalanobis_distance(object a, object b, object VI=None):
    cdef np.ndarray[DTYPE_t, ndim=1] A = np.asarray(a, dtype=np.float64).reshape(-1)
    cdef np.ndarray[DTYPE_t, ndim=1] B = np.asarray(b, dtype=np.float64).reshape(-1)
    cdef np.ndarray[DTYPE_t, ndim=2] VI_mat
    cdef Py_ssize_t d = A.shape[0]
    cdef Py_ssize_t i, j
    cdef np.ndarray[DTYPE_t, ndim=1] diff = np.empty(d, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] tmp = np.empty(d, dtype=np.float64)
    cdef DTYPE_t acc = 0.0

    if A.shape[0] != B.shape[0]:
        raise ValueError("a and b must have the same shape")
    if VI is None:
        VI_mat = np.eye(d, dtype=np.float64)
    else:
        VI_mat = np.asarray(VI, dtype=np.float64)
    if VI_mat.ndim != 2 or VI_mat.shape[0] != VI_mat.shape[1]:
        raise ValueError("VI must be a square matrix")
    if VI_mat.shape[0] != d:
        raise ValueError("VI must match the number of features")

    for i in range(d):
        diff[i] = A[i] - B[i]
    for i in range(d):
        acc = 0.0
        for j in range(d):
            acc += VI_mat[i, j] * diff[j]
        tmp[i] = acc
    acc = 0.0
    for i in range(d):
        acc += diff[i] * tmp[i]
    return sqrt(acc)


def canberra_distance(object a, object b):
    cdef np.ndarray[DTYPE_t, ndim=1] A = np.asarray(a, dtype=np.float64).reshape(-1)
    cdef np.ndarray[DTYPE_t, ndim=1] B = np.asarray(b, dtype=np.float64).reshape(-1)
    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t i
    cdef DTYPE_t acc = 0.0, num, den
    if A.shape[0] != B.shape[0]:
        raise ValueError("a and b must have the same shape")
    for i in range(n):
        num = abs(A[i] - B[i])
        den = abs(A[i]) + abs(B[i])
        acc += num / (den if den > 1e-12 else 1e-12)
    return acc


def bray_curtis_distance(object a, object b):
    cdef np.ndarray[DTYPE_t, ndim=1] A = np.asarray(a, dtype=np.float64).reshape(-1)
    cdef np.ndarray[DTYPE_t, ndim=1] B = np.asarray(b, dtype=np.float64).reshape(-1)
    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t i
    cdef DTYPE_t num = 0.0, den = 0.0
    if A.shape[0] != B.shape[0]:
        raise ValueError("a and b must have the same shape")
    for i in range(n):
        num += fabs(A[i] - B[i])
        den += fabs(A[i] + B[i])
    return num / (den if den > 1e-12 else 1e-12)
