# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Cython kernels for 1D convolution packing and unpacking."""

import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t ITYPE_t

np.import_array()


def im2col1d(np.ndarray[DTYPE_t, ndim=3] X, int kernel_size, int stride=1, int pad=0):
    cdef ITYPE_t N = X.shape[0]
    cdef ITYPE_t T = X.shape[1]
    cdef ITYPE_t C = X.shape[2]
    cdef ITYPE_t OT = (T + 2 * pad - kernel_size) // stride + 1
    cdef np.ndarray[DTYPE_t, ndim=3] X_pad
    cdef np.ndarray[DTYPE_t, ndim=2] cols
    cdef DTYPE_t[:, :, :] xv
    cdef DTYPE_t[:, :] cv
    cdef ITYPE_t n, t, k, c, row, col

    if pad > 0:
        X_pad = np.pad(np.ascontiguousarray(X, dtype=np.float64), ((0, 0), (pad, pad), (0, 0)), mode="constant")
    else:
        X_pad = np.ascontiguousarray(X, dtype=np.float64)
    cols = np.empty((N * OT, kernel_size * C), dtype=np.float64)
    xv = X_pad
    cv = cols
    for n in range(N):
        for t in range(OT):
            row = n * OT + t
            for k in range(kernel_size):
                for c in range(C):
                    col = k * C + c
                    cv[row, col] = xv[n, t * stride + k, c]
    return cols, OT


def col2im1d(np.ndarray[DTYPE_t, ndim=2] cols, shape, int kernel_size, int stride=1, int pad=0):
    cdef ITYPE_t N = shape[0]
    cdef ITYPE_t T = shape[1]
    cdef ITYPE_t C = shape[2]
    cdef ITYPE_t OT = (T + 2 * pad - kernel_size) // stride + 1
    cdef np.ndarray[DTYPE_t, ndim=3] dX_pad = np.zeros((N, T + 2 * pad, C), dtype=np.float64)
    cdef DTYPE_t[:, :, :] dv = dX_pad
    cdef DTYPE_t[:, :] cv = cols
    cdef ITYPE_t n, t, k, c, row, col

    for n in range(N):
        for t in range(OT):
            row = n * OT + t
            for k in range(kernel_size):
                for c in range(C):
                    col = k * C + c
                    dv[n, t * stride + k, c] += cv[row, col]
    if pad > 0:
        return dX_pad[:, pad:-pad, :]
    return dX_pad
