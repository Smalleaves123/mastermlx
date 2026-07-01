# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Cython helpers for Kalman-style filter hot paths."""

import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t


def kalman_predict(object x, object P, object F, object Q, object B=None, object u=None):
    """Run the linear Kalman predict step."""

    cdef np.ndarray[DTYPE_t, ndim=2] x_mat = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    cdef np.ndarray[DTYPE_t, ndim=2] P_mat = np.asarray(P, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] F_mat = np.asarray(F, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] Q_mat = np.asarray(Q, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] x_pred
    cdef np.ndarray[DTYPE_t, ndim=2] P_pred

    x_pred = F_mat @ x_mat
    if u is not None:
        if B is None:
            raise ValueError("Control input provided but no control matrix B is available")
        x_pred = x_pred + np.asarray(B, dtype=np.float64) @ np.asarray(u, dtype=np.float64).reshape(-1, 1)

    P_pred = F_mat @ P_mat @ F_mat.T + Q_mat
    return x_pred.ravel(), P_pred


def kalman_update(object x, object P, object z, object H, object R):
    """Run the linear Kalman update step."""

    cdef np.ndarray[DTYPE_t, ndim=2] x_mat = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    cdef np.ndarray[DTYPE_t, ndim=2] P_mat = np.asarray(P, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] z_mat = np.asarray(z, dtype=np.float64).reshape(-1, 1)
    cdef np.ndarray[DTYPE_t, ndim=2] H_mat = np.asarray(H, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] R_mat = np.asarray(R, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] y, S, PHt, K, x_post, P_post, I

    y = z_mat - H_mat @ x_mat
    S = H_mat @ P_mat @ H_mat.T + R_mat
    PHt = P_mat @ H_mat.T
    K = np.linalg.solve(S, PHt.T).T
    x_post = x_mat + K @ y
    I = np.eye(P_mat.shape[0], dtype=np.float64)
    P_post = (I - K @ H_mat) @ P_mat
    return x_post.ravel(), P_post


def kalman_update_innovation(object x, object P, object innovation, object H, object R):
    """Run a Kalman update when the innovation vector is already computed."""

    cdef np.ndarray[DTYPE_t, ndim=2] x_mat = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    cdef np.ndarray[DTYPE_t, ndim=2] P_mat = np.asarray(P, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] y = np.asarray(innovation, dtype=np.float64).reshape(-1, 1)
    cdef np.ndarray[DTYPE_t, ndim=2] H_mat = np.asarray(H, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] R_mat = np.asarray(R, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] S, PHt, K, x_post, P_post, I

    S = H_mat @ P_mat @ H_mat.T + R_mat
    PHt = P_mat @ H_mat.T
    K = np.linalg.solve(S, PHt.T).T
    x_post = x_mat + K @ y
    I = np.eye(P_mat.shape[0], dtype=np.float64)
    P_post = (I - K @ H_mat) @ P_mat
    return x_post.ravel(), P_post
