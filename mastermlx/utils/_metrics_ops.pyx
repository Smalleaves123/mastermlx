# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Cython helpers for metric hot paths."""

import numpy as np
cimport numpy as np

ctypedef np.int64_t DTYPE_t


def confusion_matrix_counts(object y_true, object y_pred, object labels):
    cdef np.ndarray yt = np.asarray(y_true)
    cdef np.ndarray yp = np.asarray(y_pred)
    cdef np.ndarray lbls = np.asarray(labels)
    cdef Py_ssize_t n = yt.shape[0]
    cdef Py_ssize_t m = lbls.shape[0]
    cdef Py_ssize_t i
    cdef Py_ssize_t row
    cdef Py_ssize_t col
    cdef dict index = {lbls[i]: i for i in range(m)}
    cdef np.ndarray[DTYPE_t, ndim=2] cm = np.zeros((m, m), dtype=np.int64)
    cdef object yt_item
    cdef object yp_item

    if yt.ndim != 1 or yp.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays")
    if n != yp.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    for i in range(n):
        yt_item = yt[i]
        yp_item = yp[i]
        try:
            row = index[yt_item]
            col = index[yp_item]
        except KeyError as e:
            raise KeyError(e.args[0]) from None
        cm[row, col] += 1
    return cm