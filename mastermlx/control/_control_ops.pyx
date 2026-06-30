# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Cython helpers for control optimization hot loops."""

import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t


def finite_difference_jacobian(func, np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] u, double eps=1e-5):
    """Finite-difference Jacobian for a Python dynamics callback.

    Returns (A, B) where A = d f / d x and B = d f / d u.
    """

    cdef Py_ssize_t i, n = x.shape[0], m = u.shape[0]
    cdef np.ndarray fx
    cdef np.ndarray[DTYPE_t, ndim=2] A, B
    cdef np.ndarray[DTYPE_t, ndim=1] dx, du, y1, y2
    cdef Py_ssize_t out_dim

    fx = np.asarray(func(x, u), dtype=float).reshape(-1)
    out_dim = fx.shape[0]
    A = np.empty((out_dim, n), dtype=np.float64)
    B = np.empty((out_dim, m), dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    du = np.zeros(m, dtype=np.float64)

    for i in range(n):
        dx[:] = 0.0
        dx[i] = eps
        y1 = np.asarray(func(x + dx, u), dtype=float).reshape(-1)
        y2 = np.asarray(func(x - dx, u), dtype=float).reshape(-1)
        A[:, i] = (y1 - y2) / (2.0 * eps)

    for i in range(m):
        du[:] = 0.0
        du[i] = eps
        y1 = np.asarray(func(x, u + du), dtype=float).reshape(-1)
        y2 = np.asarray(func(x, u - du), dtype=float).reshape(-1)
        B[:, i] = (y1 - y2) / (2.0 * eps)

    return A, B


def rollout_dynamics(func, x0, U, dt=None, args=None):
    """Roll out a discrete or Euler-discretized continuous system."""

    cdef np.ndarray[DTYPE_t, ndim=1] x = np.asarray(x0, dtype=float).reshape(-1)
    cdef np.ndarray U_arr = np.asarray(U, dtype=float)
    cdef Py_ssize_t T = U_arr.shape[0]
    cdef Py_ssize_t t
    cdef list states = [x.copy()]
    cdef tuple extra = () if args is None else tuple(args)
    cdef np.ndarray u, next_state, dx
    cdef double dt_val = 0.0 if dt is None else float(dt)

    for t in range(T):
        u = np.asarray(U_arr[t], dtype=float).reshape(-1)
        if dt is None:
            next_state = np.asarray(func(x, u, *extra), dtype=float).reshape(-1)
        else:
            dx = np.asarray(func(x, u, *extra), dtype=float).reshape(-1)
            next_state = x + dt_val * dx
        x = np.asarray(next_state, dtype=float).reshape(-1)
        states.append(x.copy())

    return np.asarray(states)


def quadratic_trajectory_cost(X, U, Q, R, Qf, x_ref=None, u_ref=None):
    """Compute quadratic trajectory cost for a rollout."""

    cdef np.ndarray X_arr = np.asarray(X, dtype=float)
    cdef np.ndarray U_arr = np.asarray(U, dtype=float)
    cdef np.ndarray Qm = np.asarray(Q, dtype=float)
    cdef np.ndarray Rm = np.asarray(R, dtype=float)
    cdef np.ndarray Qfm = np.asarray(Qf, dtype=float)
    cdef object x_ref_arr = None if x_ref is None else np.asarray(x_ref, dtype=float)
    cdef object u_ref_arr = None if u_ref is None else np.asarray(u_ref, dtype=float)
    cdef Py_ssize_t T = U_arr.shape[0]
    cdef Py_ssize_t t
    cdef np.ndarray x, u, xr, ur, dx, du
    cdef double cost = 0.0

    for t in range(T):
        x = np.asarray(X_arr[t], dtype=float).reshape(-1)
        u = np.asarray(U_arr[t], dtype=float).reshape(-1)
        xr = np.zeros_like(x) if x_ref_arr is None else np.asarray(x_ref_arr[t], dtype=float).reshape(-1)
        ur = np.zeros_like(u) if u_ref_arr is None else np.asarray(u_ref_arr[t], dtype=float).reshape(-1)
        dx = x - xr
        du = u - ur
        cost += float(dx @ Qm @ dx + du @ Rm @ du)

    x = np.asarray(X_arr[T], dtype=float).reshape(-1)
    xr = np.zeros_like(x) if x_ref_arr is None else np.asarray(x_ref_arr[T], dtype=float).reshape(-1)
    dx = x - xr
    cost += float(dx @ Qfm @ dx)
    return cost
