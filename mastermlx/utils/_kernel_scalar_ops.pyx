# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Cython helpers for scalar kernel hot paths."""

import numpy as np
cimport numpy as np
from libc.math cimport cosh, tanh

ctypedef np.float64_t DTYPE_t


def linear_kernel(object X, object Y):
    cdef np.ndarray[DTYPE_t, ndim=2] X_arr = np.asarray(X, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] Y_arr = np.asarray(Y, dtype=np.float64)
    return X_arr @ Y_arr.T


def cosine_kernel(object X, object Y):
    cdef np.ndarray[DTYPE_t, ndim=2] X_arr = np.asarray(X, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] Y_arr = np.asarray(Y, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] out = np.empty((X_arr.shape[0], Y_arr.shape[0]), dtype=np.float64)
    cdef Py_ssize_t i, j, k, d = X_arr.shape[1]
    cdef DTYPE_t dot, xnorm, ynorm, denom

    for i in range(X_arr.shape[0]):
        for j in range(Y_arr.shape[0]):
            dot = 0.0
            xnorm = 0.0
            ynorm = 0.0
            for k in range(d):
                dot += X_arr[i, k] * Y_arr[j, k]
                xnorm += X_arr[i, k] * X_arr[i, k]
                ynorm += Y_arr[j, k] * Y_arr[j, k]
            denom = (xnorm ** 0.5) * (ynorm ** 0.5)
            out[i, j] = dot / (denom if denom > 1e-12 else 1e-12)
    return out


def poly_kernel(object X, object Y, double gamma, double coef0, int degree):
    cdef np.ndarray[DTYPE_t, ndim=2] K = np.asarray(X, dtype=np.float64) @ np.asarray(Y, dtype=np.float64).T
    return (gamma * K + coef0) ** degree


def sigmoid_kernel(object X, object Y, double gamma, double coef0):
    cdef np.ndarray[DTYPE_t, ndim=2] K = np.asarray(X, dtype=np.float64) @ np.asarray(Y, dtype=np.float64).T
    return np.tanh(gamma * K + coef0)
