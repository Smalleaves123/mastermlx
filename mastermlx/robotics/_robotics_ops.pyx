# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Cython helpers for robotics kinematics hot paths."""

from libc.math cimport cos, sin

import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.int8_t ITYPE_t


cdef inline void _matmul4(np.ndarray[DTYPE_t, ndim=2] A, np.ndarray[DTYPE_t, ndim=2] B, np.ndarray[DTYPE_t, ndim=2] out):
    cdef int r, c
    for r in range(4):
        for c in range(4):
            out[r, c] = (
                A[r, 0] * B[0, c]
                + A[r, 1] * B[1, c]
                + A[r, 2] * B[2, c]
                + A[r, 3] * B[3, c]
            )


cdef inline np.ndarray[DTYPE_t, ndim=2] _as_transform(object T):
    cdef np.ndarray[DTYPE_t, ndim=2] arr = np.asarray(T, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != 4 or arr.shape[1] != 4:
        raise ValueError("Expected a 4x4 transform")
    return arr


def forward_kinematics_dh(
    np.ndarray[DTYPE_t, ndim=1] a,
    np.ndarray[DTYPE_t, ndim=1] alpha,
    np.ndarray[DTYPE_t, ndim=1] d,
    np.ndarray[DTYPE_t, ndim=1] theta,
    np.ndarray[ITYPE_t, ndim=1] joint_type,
    np.ndarray[DTYPE_t, ndim=1] offset,
    np.ndarray[DTYPE_t, ndim=1] q,
    base=None,
    tool=None,
    bint return_frames=True,
):
    """Compute forward kinematics for packed DH parameters."""

    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t i
    cdef double qi, ai, alphai, di, thetai, offi, ct, st, ca, sa
    cdef np.ndarray[DTYPE_t, ndim=2] T = np.asarray(np.eye(4, dtype=np.float64) if base is None else base, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] A = np.empty((4, 4), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] T_new = np.empty((4, 4), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] T_tmp
    cdef np.ndarray[DTYPE_t, ndim=2] tool_arr = None
    cdef np.ndarray[DTYPE_t, ndim=3] frames = None

    if T.ndim != 2 or T.shape[0] != 4 or T.shape[1] != 4:
        raise ValueError("Expected a 4x4 base transform")
    if tool is not None:
        tool_arr = _as_transform(tool)

    if return_frames:
        frames = np.empty((n + 1 + (1 if tool_arr is not None else 0), 4, 4), dtype=np.float64)
        frames[0] = T

    for i in range(n):
        qi = q[i]
        ai = a[i]
        alphai = alpha[i]
        di = d[i]
        thetai = theta[i]
        offi = offset[i]

        if joint_type[i] != 0:
            thetai = thetai + qi + offi
        else:
            di = di + qi + offi

        ct = cos(thetai)
        st = sin(thetai)
        ca = cos(alphai)
        sa = sin(alphai)

        A[0, 0] = ct
        A[0, 1] = -st * ca
        A[0, 2] = st * sa
        A[0, 3] = ai * ct
        A[1, 0] = st
        A[1, 1] = ct * ca
        A[1, 2] = -ct * sa
        A[1, 3] = ai * st
        A[2, 0] = 0.0
        A[2, 1] = sa
        A[2, 2] = ca
        A[2, 3] = di
        A[3, 0] = 0.0
        A[3, 1] = 0.0
        A[3, 2] = 0.0
        A[3, 3] = 1.0

        _matmul4(T, A, T_new)
        T_tmp = T
        T = T_new
        T_new = T_tmp
        if return_frames:
            frames[i + 1] = T

    if tool_arr is not None:
        _matmul4(T, tool_arr, T_new)
        T_tmp = T
        T = T_new
        T_new = T_tmp
        if return_frames:
            frames[n + 1] = T

    if return_frames:
        return T, frames
    return T


def geometric_jacobian_dh(
    np.ndarray[DTYPE_t, ndim=1] a,
    np.ndarray[DTYPE_t, ndim=1] alpha,
    np.ndarray[DTYPE_t, ndim=1] d,
    np.ndarray[DTYPE_t, ndim=1] theta,
    np.ndarray[ITYPE_t, ndim=1] joint_type,
    np.ndarray[DTYPE_t, ndim=1] offset,
    np.ndarray[DTYPE_t, ndim=1] q,
    base=None,
    tool=None,
):
    """Compute the geometric Jacobian for packed DH parameters."""

    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t i
    cdef double qi, ai, alphai, di, thetai, offi, ct, st, ca, sa
    cdef np.ndarray[DTYPE_t, ndim=2] T = np.asarray(np.eye(4, dtype=np.float64) if base is None else base, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] A = np.empty((4, 4), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] T_new = np.empty((4, 4), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] T_tmp
    cdef np.ndarray[DTYPE_t, ndim=2] tool_arr = None
    cdef np.ndarray[DTYPE_t, ndim=2] J = np.zeros((6, n), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] origins = np.empty((n, 3), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] axes = np.empty((n, 3), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] p_end

    if T.ndim != 2 or T.shape[0] != 4 or T.shape[1] != 4:
        raise ValueError("Expected a 4x4 base transform")
    if tool is not None:
        tool_arr = _as_transform(tool)

    for i in range(n):
        origins[i, 0] = T[0, 3]
        origins[i, 1] = T[1, 3]
        origins[i, 2] = T[2, 3]
        axes[i, 0] = T[0, 2]
        axes[i, 1] = T[1, 2]
        axes[i, 2] = T[2, 2]

        qi = q[i]
        ai = a[i]
        alphai = alpha[i]
        di = d[i]
        thetai = theta[i]
        offi = offset[i]

        if joint_type[i] != 0:
            thetai = thetai + qi + offi
        else:
            di = di + qi + offi

        ct = cos(thetai)
        st = sin(thetai)
        ca = cos(alphai)
        sa = sin(alphai)

        A[0, 0] = ct
        A[0, 1] = -st * ca
        A[0, 2] = st * sa
        A[0, 3] = ai * ct
        A[1, 0] = st
        A[1, 1] = ct * ca
        A[1, 2] = -ct * sa
        A[1, 3] = ai * st
        A[2, 0] = 0.0
        A[2, 1] = sa
        A[2, 2] = ca
        A[2, 3] = di
        A[3, 0] = 0.0
        A[3, 1] = 0.0
        A[3, 2] = 0.0
        A[3, 3] = 1.0

        _matmul4(T, A, T_new)
        T_tmp = T
        T = T_new
        T_new = T_tmp

    if tool_arr is not None:
        _matmul4(T, tool_arr, T_new)
        T_tmp = T
        T = T_new
        T_new = T_tmp

    p_end = T[:, 3]
    for i in range(n):
        if joint_type[i] != 0:
            J[0, i] = axes[i, 1] * (p_end[2] - origins[i, 2]) - axes[i, 2] * (p_end[1] - origins[i, 1])
            J[1, i] = axes[i, 2] * (p_end[0] - origins[i, 0]) - axes[i, 0] * (p_end[2] - origins[i, 2])
            J[2, i] = axes[i, 0] * (p_end[1] - origins[i, 1]) - axes[i, 1] * (p_end[0] - origins[i, 0])
            J[3, i] = axes[i, 0]
            J[4, i] = axes[i, 1]
            J[5, i] = axes[i, 2]
        else:
            J[0, i] = axes[i, 0]
            J[1, i] = axes[i, 1]
            J[2, i] = axes[i, 2]
            J[3, i] = 0.0
            J[4, i] = 0.0
            J[5, i] = 0.0

    return J
