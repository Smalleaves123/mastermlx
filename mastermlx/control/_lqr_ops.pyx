# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Cython helpers for Riccati-style control hot paths."""

import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t


def solve_discrete_are(np.ndarray[DTYPE_t, ndim=2] A, np.ndarray[DTYPE_t, ndim=2] B, np.ndarray[DTYPE_t, ndim=2] Q, np.ndarray[DTYPE_t, ndim=2] R, int max_iter=1000, double tol=1e-9):
    """Solve the discrete algebraic Riccati equation by fixed-point iteration."""

    cdef np.ndarray[DTYPE_t, ndim=2] P = np.asarray(Q, dtype=np.float64).copy()
    cdef np.ndarray[DTYPE_t, ndim=2] BtP, S, K, P_next, diff
    cdef Py_ssize_t i

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    for i in range(max_iter):
        BtP = B.T @ P
        S = R + BtP @ B
        K = np.linalg.solve(S, BtP @ A)
        P_next = A.T @ P @ A - A.T @ P @ B @ K + Q
        diff = np.abs(P_next - P)
        if np.max(diff) <= tol:
            return P_next
        P = P_next
    return P


def finite_horizon_lqr(np.ndarray[DTYPE_t, ndim=2] A, np.ndarray[DTYPE_t, ndim=2] B, np.ndarray[DTYPE_t, ndim=2] Q, np.ndarray[DTYPE_t, ndim=2] R, int horizon, Qf=None, reference=None):
    """Finite-horizon discrete LQR solved by Riccati backward recursion."""

    cdef Py_ssize_t t
    cdef np.ndarray[DTYPE_t, ndim=2] S, K_t
    cdef list P, K

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if Qf is None:
        Qf = Q
    Qf = np.asarray(Qf, dtype=np.float64)
    reference = None if reference is None else np.asarray(reference, dtype=np.float64)

    P = [None] * (horizon + 1)
    K = [None] * horizon
    P[horizon] = Qf.copy()

    for t in range(horizon - 1, -1, -1):
        S = R + B.T @ P[t + 1] @ B
        K_t = np.linalg.solve(S, B.T @ P[t + 1] @ A)
        K[t] = K_t
        P[t] = Q + A.T @ P[t + 1] @ (A - B @ K_t)
    return K, P, reference
