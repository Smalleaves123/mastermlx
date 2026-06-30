# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Cython accelerated decision tree split search.

Optimized version:
  - Replaces Python dict class-to-index lookups with direct array indexing.
  - Uses typed memoryviews everywhere; avoids X[:, j] slicing.
  - Single np.argsort per feature column (amortized for the inner scan).
"""
import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t   ITYPE_t
ctypedef np.int64_t  INT_t


cdef inline ITYPE_t _get_class_idx(INT_t cls, INT_t[:] classes, ITYPE_t[:] lookup,
                                    ITYPE_t offset, ITYPE_t n_classes) noexcept:
    """O(1) class-label → index via direct lookup when labels are dense."""
    cdef ITYPE_t pos = cls - offset
    if 0 <= pos < lookup.shape[0]:
        return lookup[pos]
    # Fallback — should not happen if lookup is built correctly
    cdef ITYPE_t lo = 0, hi = n_classes - 1, mid
    while lo <= hi:
        mid = (lo + hi) // 2
        if classes[mid] < cls:
            lo = mid + 1
        elif classes[mid] > cls:
            hi = mid - 1
        else:
            return mid
    return -1


def _best_split_classifier(
    np.ndarray[DTYPE_t, ndim=2] X,
    np.ndarray[INT_t, ndim=1] y,
    int min_samples_leaf,
):
    """Find best (feature, threshold) for a classification node (Gini impurity).

    Returns (best_feat, best_thr, best_score) or (None, None, None).
    """
    cdef ITYPE_t n = X.shape[0]
    cdef ITYPE_t m = X.shape[1]

    if n < 2 * min_samples_leaf:
        return None, None, None

    # ---- typed memoryviews -----
    cdef DTYPE_t[:, :] Xv = X
    cdef INT_t[:]      yv = y

    # ---- class map (dense lookup, not dict) ----
    cdef np.ndarray[INT_t, ndim=1] classes = np.unique(y)
    cdef ITYPE_t n_classes = classes.shape[0]
    if n_classes <= 1:
        return None, None, None

    cdef INT_t min_cls = classes[0]
    cdef INT_t max_cls = classes[n_classes - 1]
    cdef ITYPE_t lookup_span = max_cls - min_cls + 1

    cdef np.ndarray[ITYPE_t, ndim=1] class_lookup
    cdef ITYPE_t[:] lookup_v
    cdef ITYPE_t offset = min_cls
    cdef ITYPE_t c

    if lookup_span <= 100000:          # dense labels → direct O(1)
        class_lookup = np.full(lookup_span, -1, dtype=np.intp)
        for c in range(n_classes):
            class_lookup[classes[c] - min_cls] = c
        lookup_v = class_lookup
    else:
        class_lookup = np.empty(0, dtype=np.intp)
        lookup_v = class_lookup

    # ---- total class counts ----
    cdef np.ndarray[DTYPE_t, ndim=1] total_counts = np.zeros(n_classes, dtype=np.float64)
    cdef ITYPE_t i, j, k
    cdef INT_t cls_val

    for i in range(n):
        cls_val = yv[i]
        total_counts[_get_class_idx(cls_val, classes, lookup_v, offset, n_classes)] += 1.0

    cdef DTYPE_t total_sum = <DTYPE_t>n

    # Check if pure node
    cdef DTYPE_t base_gini = 1.0
    for c in range(n_classes):
        base_gini -= (total_counts[c] / total_sum) ** 2
    if base_gini < 1e-15:
        return None, None, None

    # ---- split search ----
    cdef DTYPE_t best_score = 1e308
    cdef ITYPE_t best_feat = -1
    cdef DTYPE_t best_thr = 0.0

    cdef np.ndarray[DTYPE_t, ndim=1] left_counts = np.zeros(n_classes, dtype=np.float64)
    cdef DTYPE_t[:] left_cv = left_counts
    cdef np.ndarray[ITYPE_t, ndim=1] order
    cdef ITYPE_t[:] ord_v
    cdef np.ndarray[DTYPE_t, ndim=1] x_col = np.empty(n, dtype=np.float64)
    cdef DTYPE_t[:] xcv = x_col

    cdef DTYPE_t left_n, right_n, left_gini, right_gini, score
    cdef DTYPE_t current_val, next_val
    cdef ITYPE_t row_cur, row_nxt

    for j in range(m):
        # copy column into typed buffer (avoid repeated X[:, j] slicing)
        for i in range(n):
            xcv[i] = Xv[i, j]

        order = np.argsort(x_col)              # still a Python call, but once per feature
        ord_v = order

        # reset counts
        for c in range(n_classes):
            left_cv[c] = 0.0
        left_n = 0.0

        for i in range(n - 1):
            row_cur = ord_v[i]
            cls_val = yv[row_cur]
            left_cv[_get_class_idx(cls_val, classes, lookup_v, offset, n_classes)] += 1.0
            left_n += 1.0
            right_n = total_sum - left_n

            # skip equal values (no threshold possible)
            current_val = xcv[row_cur]
            next_val = xcv[ord_v[i + 1]]
            if current_val == next_val:
                continue
            if left_n < min_samples_leaf or right_n < min_samples_leaf:
                continue

            # left Gini
            left_gini = 1.0
            for c in range(n_classes):
                left_gini -= (left_cv[c] / left_n) ** 2

            # right Gini
            right_gini = 1.0
            for c in range(n_classes):
                right_gini -= ((total_counts[c] - left_cv[c]) / right_n) ** 2

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
    """Find best (feature, threshold) for a regression node (MSE reduction).

    Returns (best_feat, best_thr, best_score) or (None, None, None).
    """
    cdef ITYPE_t n = X.shape[0]
    cdef ITYPE_t m = X.shape[1]

    if n < 2 * min_samples_leaf:
        return None, None, None

    cdef DTYPE_t[:, :] Xv = X
    cdef DTYPE_t[:]   yv = y

    # ---- pre-compute totals ----
    cdef DTYPE_t total_sum_y = 0.0
    cdef DTYPE_t total_sum_y2 = 0.0
    cdef ITYPE_t i, j

    for i in range(n):
        total_sum_y += yv[i]
        total_sum_y2 += yv[i] * yv[i]

    cdef DTYPE_t total_n = <DTYPE_t>n
    cdef DTYPE_t base_var = total_sum_y2 / total_n - (total_sum_y / total_n) ** 2
    if base_var < 1e-15:
        return None, None, None

    cdef DTYPE_t best_score = 1e308
    cdef ITYPE_t best_feat = -1
    cdef DTYPE_t best_thr = 0.0

    cdef np.ndarray[ITYPE_t, ndim=1] order
    cdef ITYPE_t[:] ord_v
    cdef np.ndarray[DTYPE_t, ndim=1] x_col = np.empty(n, dtype=np.float64)
    cdef DTYPE_t[:] xcv = x_col

    cdef DTYPE_t left_sum_y, left_sum_y2, right_sum_y, right_sum_y2
    cdef DTYPE_t left_n, right_n, left_var, right_var, score
    cdef DTYPE_t current_val, next_val
    cdef ITYPE_t row_cur

    for j in range(m):
        for i in range(n):
            xcv[i] = Xv[i, j]

        order = np.argsort(x_col)
        ord_v = order

        left_sum_y = 0.0
        left_sum_y2 = 0.0
        left_n = 0.0

        for i in range(n - 1):
            row_cur = ord_v[i]
            left_sum_y += yv[row_cur]
            left_sum_y2 += yv[row_cur] * yv[row_cur]
            left_n += 1.0
            right_n = total_n - left_n

            current_val = xcv[row_cur]
            next_val = xcv[ord_v[i + 1]]
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
