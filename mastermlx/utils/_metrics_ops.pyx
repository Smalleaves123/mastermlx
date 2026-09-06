# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Cython helpers for metric hot paths."""

import numpy as np
cimport numpy as np

ctypedef np.int64_t DTYPE_t
ctypedef np.float64_t FLOAT_t


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


def top_k_accuracy(
    np.ndarray[DTYPE_t, ndim=1] true_indices,
    np.ndarray[FLOAT_t, ndim=2] scores,
    int k,
):
    """Return top-k accuracy without materializing a sorted score matrix."""

    cdef Py_ssize_t n = scores.shape[0]
    cdef Py_ssize_t n_classes = scores.shape[1]
    cdef Py_ssize_t sample, column, target
    cdef Py_ssize_t better
    cdef Py_ssize_t correct = 0
    cdef double target_score

    if true_indices.shape[0] != n:
        raise ValueError("true_indices and scores must have the same number of rows")
    if n == 0:
        raise ValueError("top-k accuracy requires at least one sample")
    if k < 1 or k > n_classes:
        raise ValueError("k must be between 1 and the number of classes")

    for sample in range(n):
        target = true_indices[sample]
        if target < 0 or target >= n_classes:
            raise ValueError("true_indices contains an invalid class index")
        target_score = scores[sample, target]
        better = 0
        for column in range(n_classes):
            if (
                scores[sample, column] > target_score
                or (scores[sample, column] == target_score and column > target)
            ):
                better += 1
                if better >= k:
                    break
        if better < k:
            correct += 1
    return correct / <double>n
