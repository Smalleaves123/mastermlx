# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Cython forward kernel for a tanh recurrent layer."""

import numpy as np
cimport numpy as np
from libc.math cimport tanh

ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t ITYPE_t

np.import_array()


def simple_rnn_forward(np.ndarray[DTYPE_t, ndim=3] X,
                       np.ndarray[DTYPE_t, ndim=2] W_xh,
                       np.ndarray[DTYPE_t, ndim=2] W_hh,
                       np.ndarray[DTYPE_t, ndim=1] b):
    cdef ITYPE_t N = X.shape[0]
    cdef ITYPE_t T = X.shape[1]
    cdef ITYPE_t D = X.shape[2]
    cdef ITYPE_t U = W_hh.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=3] H = np.zeros((N, T, U), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] h = np.zeros((N, U), dtype=np.float64)
    cdef ITYPE_t n, t, d, u, v
    cdef double value

    for t in range(T):
        for n in range(N):
            for u in range(U):
                value = b[u]
                for d in range(D):
                    value += X[n, t, d] * W_xh[d, u]
                for v in range(U):
                    value += h[n, v] * W_hh[v, u]
                h[n, u] = tanh(value)
                H[n, t, u] = h[n, u]
    return H
