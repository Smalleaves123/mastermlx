from __future__ import annotations

from math import comb as _comb

import numpy as np

from .distance import pairwise_distance


def _check_labels(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty 1D array")
    return labels


def _pairwise_dists(X):
    return pairwise_distance(X, X, metric="euclidean")


# ---------------------------------------------------------------------------
# Internal cluster metrics
# ---------------------------------------------------------------------------


def silhouette(X, labels):
    X = np.asarray(X, dtype=float)
    labels = _check_labels(labels)
    if X.shape[0] != labels.size:
        raise ValueError("X and labels must have the same number of rows")
    uniq = np.unique(labels)
    k = uniq.size
    if k < 2 or k == labels.size:
        raise ValueError("silhouette requires at least 2 clusters and fewer clusters than samples")

    dists = _pairwise_dists(X)
    scores = np.empty(labels.size, dtype=float)

    for i in range(labels.size):
        li = labels[i]
        same = labels == li
        other = labels != li
        if np.sum(same) <= 1:
            a_i = 0.0
        else:
            a_i = np.mean(dists[i, same])
        b_i = np.inf
        for c in uniq:
            if c == li:
                continue
            b_i = min(b_i, np.mean(dists[i, labels == c]))
        scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0

    return float(np.mean(scores))


def davies_bouldin(X, labels):
    X = np.asarray(X, dtype=float)
    labels = _check_labels(labels)
    if X.shape[0] != labels.size:
        raise ValueError("X and labels must have the same number of rows")
    uniq = np.unique(labels)
    k = uniq.size
    if k < 2:
        raise ValueError("davies_bouldin requires at least 2 clusters")

    centroids = np.array([np.mean(X[labels == c], axis=0) for c in uniq])
    scatter = np.array([np.mean(np.linalg.norm(X[labels == c] - centroids[i], axis=1))
                         for i, c in enumerate(uniq)])
    score = 0.0
    for i in range(k):
        ri = 0.0
        for j in range(k):
            if i == j:
                continue
            d_ij = np.linalg.norm(centroids[i] - centroids[j])
            r = (scatter[i] + scatter[j]) / max(d_ij, 1e-12)
            ri = max(ri, r)
        score += ri
    return float(score / k)


def calinski_harabasz(X, labels):
    X = np.asarray(X, dtype=float)
    labels = _check_labels(labels)
    if X.shape[0] != labels.size:
        raise ValueError("X and labels must have the same number of rows")
    n, d = X.shape
    uniq = np.unique(labels)
    k = uniq.size
    if k < 2:
        raise ValueError("calinski_harabasz requires at least 2 clusters")

    global_center = np.mean(X, axis=0)
    ssb = 0.0
    ssw = 0.0
    for c in uniq:
        mask = labels == c
        n_k = np.sum(mask)
        center_k = np.mean(X[mask], axis=0)
        ssb += n_k * np.sum((center_k - global_center) ** 2)
        ssw += np.sum((X[mask] - center_k) ** 2)
    return float((ssb / (k - 1)) / (ssw / (n - k))) if ssw > 0 else 0.0


# ---------------------------------------------------------------------------
# External cluster metrics
# ---------------------------------------------------------------------------


def _contingency(a, b):
    uniq_a = np.unique(a)
    uniq_b = np.unique(b)
    table = np.zeros((uniq_a.size, uniq_b.size), dtype=float)
    for i, va in enumerate(uniq_a):
        for j, vb in enumerate(uniq_b):
            table[i, j] = np.sum((a == va) & (b == vb))
    return table


def _comb2(x):
    """n choose 2, safe for numpy arrays."""
    x = np.asarray(x, dtype=float)
    return x * (x - 1.0) / 2.0


def adj_rand(a, b):
    a = _check_labels(a)
    b = _check_labels(b)
    if a.size != b.size:
        raise ValueError("a and b must have the same length")
    ct = _contingency(a, b)
    n = a.size
    row_sum = ct.sum(axis=1)
    col_sum = ct.sum(axis=0)
    sum_comb = float(np.sum(_comb2(ct)))
    row_comb = float(np.sum(_comb2(row_sum)))
    col_comb = float(np.sum(_comb2(col_sum)))
    total_comb = float(_comb2(n))
    exp = row_comb * col_comb / max(total_comb, 1.0)
    max_idx = (row_comb + col_comb) / 2.0
    den = max_idx - exp
    return float((sum_comb - exp) / den) if den > 0 else 0.0


def _entropy_vec(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def adj_mi(a, b):
    a = _check_labels(a)
    b = _check_labels(b)
    if a.size != b.size:
        raise ValueError("a and b must have the same length")
    ct = _contingency(a, b)
    n = int(a.size)
    row_sum = ct.sum(axis=1).astype(int)
    col_sum = ct.sum(axis=0).astype(int)

    # Mutual information
    mi = 0.0
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            nij = int(ct[i, j])
            if nij > 0:
                mi += (nij / n) * np.log(n * nij / (row_sum[i] * col_sum[j]))

    # Expected MI (hypergeometric model)
    emi = 0.0
    for i in range(ct.shape[0]):
        ai = int(row_sum[i])
        for j in range(ct.shape[1]):
            bj = int(col_sum[j])
            lo = max(1, ai + bj - n)
            hi = min(ai, bj)
            total = 0.0
            for k in range(lo, hi + 1):
                total += _comb(ai, k) * _comb(n - ai, bj - k) / _comb(n, bj)
            term = 0.0
            for k in range(lo, hi + 1):
                p = _comb(ai, k) * _comb(n - ai, bj - k) / _comb(n, bj)
                if p > 0 and total > 0:
                    q = p / total
                    term += q * (k / n) * np.log(n * k / (ai * bj)) if k > 0 else 0.0
            emi += term * total if total > 0 else 0.0

    p_a = row_sum / n
    p_b = col_sum / n
    ha = _entropy_vec(p_a)
    hb = _entropy_vec(p_b)
    den = max(ha, hb) - emi
    return float((mi - emi) / den) if den > 0 else 0.0


def v_measure(a, b):
    a = _check_labels(a)
    b = _check_labels(b)
    if a.size != b.size:
        raise ValueError("a and b must have the same length")
    ct = _contingency(a, b)
    n = a.size
    p_ab = ct / n
    p_a = p_ab.sum(axis=1)
    p_b = p_ab.sum(axis=0)

    mi = 0.0
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            if p_ab[i, j] > 0:
                mi += p_ab[i, j] * np.log(p_ab[i, j] / (p_a[i] * p_b[j]))

    ha = -np.sum(p_a[p_a > 0] * np.log(p_a[p_a > 0]))
    hb = -np.sum(p_b[p_b > 0] * np.log(p_b[p_b > 0]))
    h = float(mi / ha) if ha > 0 else 0.0
    c = float(mi / hb) if hb > 0 else 0.0
    v = float(2.0 * h * c / (h + c)) if (h + c) > 0 else 0.0
    return h, c, v
