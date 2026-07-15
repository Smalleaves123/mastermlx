# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Cython forward kernels for recurrent layers."""

import numpy as np
cimport numpy as np
from libc.math cimport exp, tanh

ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t ITYPE_t

np.import_array()


cdef inline double _sigmoid(double value):
    """Evaluate sigmoid without overflowing for large inputs."""
    cdef double exp_value
    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    exp_value = exp(value)
    return exp_value / (1.0 + exp_value)


def simple_rnn_forward(np.ndarray[DTYPE_t, ndim=3] X,
                       np.ndarray[DTYPE_t, ndim=2] W_xh,
                       np.ndarray[DTYPE_t, ndim=2] W_hh,
                       np.ndarray[DTYPE_t, ndim=1] b):
    cdef ITYPE_t N = X.shape[0]
    cdef ITYPE_t T = X.shape[1]
    cdef ITYPE_t D = X.shape[2]
    cdef ITYPE_t U = W_hh.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=3] H = np.zeros((N, T, U), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] h_prev = np.zeros((N, U), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] h_next = np.zeros((N, U), dtype=np.float64)
    cdef ITYPE_t n, t, d, u, v
    cdef double value

    for t in range(T):
        for n in range(N):
            for u in range(U):
                value = b[u]
                for d in range(D):
                    value += X[n, t, d] * W_xh[d, u]
                for v in range(U):
                    value += h_prev[n, v] * W_hh[v, u]
                h_next[n, u] = tanh(value)
                H[n, t, u] = h_next[n, u]
        for n in range(N):
            for u in range(U):
                h_prev[n, u] = h_next[n, u]
    return H


def lstm_forward(np.ndarray[DTYPE_t, ndim=3] X,
                 np.ndarray[DTYPE_t, ndim=2] W,
                 np.ndarray[DTYPE_t, ndim=2] U,
                 np.ndarray[DTYPE_t, ndim=1] b,
                 int units):
    """Run the project's f/i/g/o LSTM recurrence."""
    cdef ITYPE_t N = X.shape[0]
    cdef ITYPE_t T = X.shape[1]
    cdef ITYPE_t D = X.shape[2]
    cdef np.ndarray[DTYPE_t, ndim=3] H = np.zeros((N, T, units), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=3] C = np.zeros((N, T, units), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=3] G = np.zeros((N, T, 4 * units), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] h_prev = np.zeros((N, units), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] h_next = np.zeros((N, units), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] c_prev = np.zeros((N, units), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] c_next = np.zeros((N, units), dtype=np.float64)
    cdef ITYPE_t n, t, d, u, v
    cdef double value, f, i, g, o, cell

    if units <= 0:
        raise ValueError("units must be positive")

    for t in range(T):
        for n in range(N):
            for u in range(units):
                value = b[u]
                for d in range(D):
                    value += X[n, t, d] * W[d, u]
                for v in range(units):
                    value += h_prev[n, v] * U[v, u]
                f = _sigmoid(value)

                value = b[units + u]
                for d in range(D):
                    value += X[n, t, d] * W[d, units + u]
                for v in range(units):
                    value += h_prev[n, v] * U[v, units + u]
                i = _sigmoid(value)

                value = b[2 * units + u]
                for d in range(D):
                    value += X[n, t, d] * W[d, 2 * units + u]
                for v in range(units):
                    value += h_prev[n, v] * U[v, 2 * units + u]
                g = tanh(value)

                value = b[3 * units + u]
                for d in range(D):
                    value += X[n, t, d] * W[d, 3 * units + u]
                for v in range(units):
                    value += h_prev[n, v] * U[v, 3 * units + u]
                o = _sigmoid(value)

                cell = f * c_prev[n, u] + i * g
                c_next[n, u] = cell
                h_next[n, u] = o * tanh(cell)
                C[n, t, u] = cell
                H[n, t, u] = h_next[n, u]
                G[n, t, u] = f
                G[n, t, units + u] = i
                G[n, t, 2 * units + u] = g
                G[n, t, 3 * units + u] = o
        for n in range(N):
            for u in range(units):
                h_prev[n, u] = h_next[n, u]
                c_prev[n, u] = c_next[n, u]
    return H, C, G


def gru_forward(np.ndarray[DTYPE_t, ndim=3] X,
                np.ndarray[DTYPE_t, ndim=2] W_zr,
                np.ndarray[DTYPE_t, ndim=2] W_h,
                np.ndarray[DTYPE_t, ndim=2] U_zr,
                np.ndarray[DTYPE_t, ndim=2] U_h,
                np.ndarray[DTYPE_t, ndim=1] b_zr,
                np.ndarray[DTYPE_t, ndim=1] b_h,
                int units):
    """Run the project's z/r/h GRU recurrence."""
    cdef ITYPE_t N = X.shape[0]
    cdef ITYPE_t T = X.shape[1]
    cdef ITYPE_t D = X.shape[2]
    cdef np.ndarray[DTYPE_t, ndim=3] H = np.zeros((N, T, units), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=3] G = np.zeros((N, T, 3 * units), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] h_prev = np.zeros((N, units), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] h_next = np.zeros((N, units), dtype=np.float64)
    cdef ITYPE_t n, t, d, u, v
    cdef double value, z, r, h_tilde

    if units <= 0:
        raise ValueError("units must be positive")

    for t in range(T):
        for n in range(N):
            for u in range(units):
                value = b_zr[u]
                for d in range(D):
                    value += X[n, t, d] * W_zr[d, u]
                for v in range(units):
                    value += h_prev[n, v] * U_zr[v, u]
                z = _sigmoid(value)

                value = b_zr[units + u]
                for d in range(D):
                    value += X[n, t, d] * W_zr[d, units + u]
                for v in range(units):
                    value += h_prev[n, v] * U_zr[v, units + u]
                r = _sigmoid(value)
                G[n, t, u] = z
                G[n, t, units + u] = r

            for u in range(units):
                value = b_h[u]
                for d in range(D):
                    value += X[n, t, d] * W_h[d, u]
                for v in range(units):
                    value += G[n, t, units + v] * h_prev[n, v] * U_h[v, u]
                h_tilde = tanh(value)
                h_next[n, u] = (1.0 - G[n, t, u]) * h_prev[n, u] + G[n, t, u] * h_tilde
                G[n, t, 2 * units + u] = h_tilde
                H[n, t, u] = h_next[n, u]
        for n in range(N):
            for u in range(units):
                h_prev[n, u] = h_next[n, u]
    return H, G
