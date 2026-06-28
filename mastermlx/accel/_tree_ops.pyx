# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython accelerated decision tree split search."""

import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t ITYPE_t
ctypedef np.int64_t INT_t


def _best_split_classifier(
    np.ndarray[DTYPE_t, ndim=2] X,
    np.ndarray[INT_t, ndim=1] y,
    int min_samples_leaf,
):
    """Find best (feature, threshold) for a classification node.

    Returns (best_feat, best_thr, best_score) or (None, None, None).
    Uses pre-sorted feature columns and cumulative counts for O(n*m) total.
    """
    cdef ITYPE_t n = X.shape[0]
    cdef ITYPE_t m = X.shape[1]
    cdef ITYPE_t i, j, k
    cdef DTYPE_t best_score = 1e308
    cdef ITYPE_t best_feat = -1
    cdef DTYPE_t best_thr = 0.0

    # Unique class labels
    cdef np.ndarray[INT_t, ndim=1] classes = np.unique(y)
    cdef ITYPE_t n_classes = classes.shape[0]
    cdef INT_t cls
    cdef dict class_to_idx = {int(classes[c]): c for c in range(n_classes)}

    # Total class counts (for Gini)
    cdef np.ndarray[DTYPE_t, ndim=1] total_counts = np.zeros(n_classes, dtype=np.float64)
    for i in range(n):
        total_counts[class_to_idx[y[i]]] += 1.0
    cdef DTYPE_t total_sum = n

    # Base Gini = 1 - sum(p_c^2)
    cdef DTYPE_t base_gini = 1.0
    for c in range(n_classes):
        base_gini -= (total_counts[c] / total_sum) ** 2

    if base_gini < 1e-15:
        return None, None, None

    cdef np.ndarray[DTYPE_t, ndim=1] left_counts = np.zeros(n_classes, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] right_counts
    cdef DTYPE_t left_n, right_n, left_gini, right_gini, score
    cdef DTYPE_t current_val, prev_val

    for j in range(m):
        # Sort feature j along with labels
        order = np.argsort(X[:, j])
        left_counts[:] = 0.0
        left_n = 0.0

        for i in range(n - 1):
            k = order[i]
            cls = class_to_idx[y[k]]
            left_counts[cls] += 1.0
            left_n += 1.0
            right_n = total_sum - left_n

            # Skip if threshold wouldn't change
            current_val = X[order[i], j]
            next_val = X[order[i + 1], j]
            if current_val == next_val:
                continue

            if left_n < min_samples_leaf or right_n < min_samples_leaf:
                continue

            # Gini left
            left_gini = 1.0
            for c in range(n_classes):
                left_gini -= (left_counts[c] / left_n) ** 2

            # Gini right = 1 - sum(((total - left) / right_n)^2)
            right_gini = 1.0
            for c in range(n_classes):
                right_gini -= ((total_counts[c] - left_counts[c]) / right_n) ** 2

            score = (left_n / total_sum) * left_gini + (right_n / total_sum) * right_gini
            if score < best_score:
                best_score = score
                best_feat = j
                best_thr = (current_val + next_val) * 0.5

    if best_feat == -1:
        return None, None, None
    return best_feat, best_thr, best_score


def _best_split_regressor(
    np.ndarray[DTYPE_t, ndim=2] X,
    np.ndarray[DTYPE_t, ndim=1] y,
    int min_samples_leaf,
):
    """Find best (feature, threshold) for a regression node (MSE)."""
    cdef ITYPE_t n = X.shape[0]
    cdef ITYPE_t m = X.shape[1]
    cdef ITYPE_t i, j
    cdef DTYPE_t best_score = 1e308
    cdef ITYPE_t best_feat = -1
    cdef DTYPE_t best_thr = 0.0

    cdef DTYPE_t total_sum_y = 0.0
    cdef DTYPE_t total_sum_y2 = 0.0
    for i in range(n):
        total_sum_y += y[i]
        total_sum_y2 += y[i] * y[i]
    cdef DTYPE_t total_n = n
    cdef DTYPE_t base_var = total_sum_y2 / total_n - (total_sum_y / total_n) ** 2

    if base_var < 1e-15:
        return None, None, None

    cdef DTYPE_t left_sum_y, left_sum_y2, right_sum_y, right_sum_y2
    cdef DTYPE_t left_n, right_n, left_var, right_var, score
    cdef DTYPE_t current_val, next_val

    for j in range(m):
        order = np.argsort(X[:, j])
        left_sum_y = 0.0
        left_sum_y2 = 0.0
        left_n = 0.0

        for i in range(n - 1):
            k = order[i]
            left_sum_y += y[k]
            left_sum_y2 += y[k] * y[k]
            left_n += 1.0
            right_n = total_n - left_n

            current_val = X[order[i], j]
            next_val = X[order[i + 1], j]
            if current_val == next_val:
                continue
            if left_n < min_samples_leaf or right_n < min_samples_leaf:
                continue

            left_var = left_sum_y2 / left_n - (left_sum_y / left_n) ** 2
            right_sum_y = total_sum_y - left_sum_y
            right_sum_y2 = total_sum_y2 - left_sum_y2
            right_var = right_sum_y2 / right_n - (right_sum_y / right_n) ** 2

            score = (left_n / total_n) * left_var + (right_n / total_n) * right_var
            if score < best_score:
                best_score = score
                best_feat = j
                best_thr = (current_val + next_val) * 0.5

    if best_feat == -1:
        return None, None, None
    return best_feat, best_thr, best_score
