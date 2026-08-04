"""Accelerated machine-learning kernels with stable NumPy fallbacks."""

from __future__ import annotations

import numpy as np

from ._validate import float_array
from .backends import _load_cpp_ml_kernels


def _matrix(value, name):
    return float_array(value, 2, name)


def _matrix_allow_nan(value, name):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if array.ndim != 2 or array.size == 0 or any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must be a non-empty 2D array")
    if np.isinf(array).any():
        raise ValueError(f"{name} must not contain infinite values")
    return np.ascontiguousarray(array)


def knn_impute(X, X_fit, n_neighbors, weights="distance"):
    """Impute missing query values from missing-aware nearest neighbors."""
    X = _matrix_allow_nan(X, "X")
    X_fit = _matrix_allow_nan(X_fit, "X_fit")
    if X.shape[1] != X_fit.shape[1]:
        raise ValueError("X and X_fit must have the same number of features")
    try:
        n_neighbors = int(n_neighbors)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_neighbors must be an integer") from exc
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1")
    if weights not in {"uniform", "distance"}:
        raise ValueError("weights must be 'uniform' or 'distance'")

    cpp = _load_cpp_ml_kernels()
    if cpp is not None:
        return cpp.knn_impute(X, X_fit, n_neighbors, weights == "distance")

    output = X.copy()
    for row in range(X.shape[0]):
        query = X[row]
        query_valid = np.isfinite(query)
        for column in np.flatnonzero(~query_valid):
            train_valid = np.isfinite(X_fit[:, column])
            candidates = np.flatnonzero(train_valid)
            if candidates.size == 0:
                continue
            common = query_valid[None, :] & np.isfinite(X_fit[candidates])
            usable = np.any(common, axis=1)
            candidates = candidates[usable]
            common = common[usable]
            if candidates.size == 0:
                continue
            diff = np.where(common, X_fit[candidates] - query[None, :], 0.0)
            distances = np.sqrt(np.sum(diff * diff, axis=1))
            order = np.argsort(distances, kind="stable")[:n_neighbors]
            values = X_fit[candidates[order], column]
            if weights == "uniform":
                output[row, column] = np.mean(values)
            else:
                selected = distances[order]
                factors = 1.0 / np.maximum(selected, 1e-12)
                output[row, column] = np.sum(factors * values) / np.sum(factors)
    return output


def rbf_affinity(X, gamma):
    X = _matrix(X, "X")
    gamma = float(gamma)
    if not np.isfinite(gamma):
        raise ValueError("gamma must be finite")
    cpp = _load_cpp_ml_kernels()
    if cpp is not None:
        return cpp.rbf_affinity(X, gamma)
    x2 = np.sum(X * X, axis=1)
    d2 = np.maximum(x2[:, None] + x2[None, :] - 2.0 * (X @ X.T), 0.0)
    affinity = np.exp(-gamma * d2)
    np.fill_diagonal(affinity, 0.0)
    return affinity


def knn_affinity(X, n_neighbors):
    X = _matrix(X, "X")
    try:
        n_neighbors = int(n_neighbors)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_neighbors must be an integer") from exc
    n = X.shape[0]
    if n_neighbors < 1 or n_neighbors >= n:
        raise ValueError("n_neighbors must be between 1 and n_samples - 1")
    cpp = _load_cpp_ml_kernels()
    if cpp is not None:
        return cpp.knn_affinity(X, n_neighbors)
    x2 = np.sum(X * X, axis=1)
    d2 = np.maximum(x2[:, None] + x2[None, :] - 2.0 * (X @ X.T), 0.0)
    d2[np.diag_indices(n)] = np.inf
    affinity = np.zeros((n, n), dtype=float)
    neighbors = np.argsort(d2, axis=1, kind="stable")[:, :n_neighbors]
    rows = np.arange(n)[:, None]
    affinity[rows, neighbors] = 1.0
    return np.maximum(affinity, affinity.T)


def _csr_from_dense_affinity(affinity):
    rows, cols = np.nonzero(affinity)
    n = affinity.shape[0]
    indptr = np.bincount(rows, minlength=n).cumsum()
    indptr = np.concatenate(([0], indptr)).astype(np.int64, copy=False)
    return indptr, cols.astype(np.int64, copy=False), affinity[rows, cols].astype(float, copy=False)


def knn_graph(X, n_neighbors):
    """Return a symmetric binary KNN graph as ``(indptr, indices, data)``."""
    X = _matrix(X, "X")
    n_neighbors = int(n_neighbors)
    n = X.shape[0]
    if n_neighbors < 1 or n_neighbors >= n:
        raise ValueError("n_neighbors must be between 1 and n_samples - 1")
    cpp = _load_cpp_ml_kernels()
    if cpp is not None:
        return cpp.knn_graph(X, n_neighbors)
    return _csr_from_dense_affinity(knn_affinity(X, n_neighbors))


def dbscan_neighbors(X, eps):
    """Return the radius-neighbor graph as CSR arrays."""
    X = _matrix(X, "X")
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")
    cpp = _load_cpp_ml_kernels()
    if cpp is not None:
        return cpp.dbscan_neighbors(X, eps)
    x2 = np.sum(X * X, axis=1)
    d2 = np.maximum(x2[:, None] + x2[None, :] - 2.0 * (X @ X.T), 0.0)
    return _csr_from_dense_affinity((d2 <= eps * eps).astype(float))


def dbscan_labels(indptr, indices, min_samples):
    """Expand DBSCAN clusters from a CSR epsilon-neighborhood graph."""
    indptr = np.asarray(indptr, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    try:
        min_samples = int(min_samples)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_samples must be an integer") from exc
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")
    if indptr.ndim != 1 or indptr.size < 2 or indices.ndim != 1:
        raise ValueError("invalid CSR arrays")
    n = indptr.size - 1
    if indptr[0] != 0 or indptr[-1] != indices.size or np.any(np.diff(indptr) < 0):
        raise ValueError("invalid CSR row pointer")
    if np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("invalid CSR indices")
    indptr = np.ascontiguousarray(indptr)
    indices = np.ascontiguousarray(indices)

    cpp = _load_cpp_ml_kernels()
    cpp_dbscan_labels = getattr(cpp, "dbscan_labels", None) if cpp is not None else None
    if callable(cpp_dbscan_labels):
        return cpp_dbscan_labels(indptr, indices, min_samples)

    core_mask = np.diff(indptr) >= min_samples
    core_samples = np.flatnonzero(core_mask).astype(np.int64, copy=False)
    labels = np.full(n, -1, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0
    for point in range(n):
        if visited[point] or not core_mask[point]:
            continue
        stack = [point]
        visited[point] = True
        labels[point] = cluster_id
        while stack:
            current = stack.pop()
            for neighbor in indices[indptr[current] : indptr[current + 1]]:
                neighbor = int(neighbor)
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
                if not visited[neighbor]:
                    visited[neighbor] = True
                    if core_mask[neighbor]:
                        stack.append(neighbor)
        cluster_id += 1
    return labels, core_samples


def meanshift_update(X, centers, bandwidth):
    """Move each center to the mean of samples within its bandwidth."""
    X = _matrix(X, "X")
    centers = _matrix(centers, "centers")
    if X.shape[1] != centers.shape[1]:
        raise ValueError("X and centers must have the same number of features")
    bandwidth = float(bandwidth)
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive and finite")
    cpp = _load_cpp_ml_kernels()
    cpp_meanshift_update = getattr(cpp, "meanshift_update", None) if cpp is not None else None
    if callable(cpp_meanshift_update):
        return cpp_meanshift_update(X, centers, bandwidth)

    updates = centers.copy()
    bandwidth_sq = bandwidth * bandwidth
    for row, center in enumerate(centers):
        distances = np.sum((X - center) ** 2, axis=1)
        mask = distances <= bandwidth_sq
        if np.any(mask):
            updates[row] = np.mean(X[mask], axis=0)
    return updates


def _hmm_inputs(sequence, start, trans, emit):
    sequence = np.asarray(sequence, dtype=np.int64)
    start = np.asarray(start, dtype=float)
    trans = _matrix(trans, "trans")
    emit = _matrix(emit, "emit")
    if sequence.ndim != 1 or sequence.size == 0 or start.ndim != 1 or start.size == 0:
        raise ValueError("sequence must be a non-empty 1D array")
    if trans.shape[0] != trans.shape[1] or start.shape[0] != trans.shape[0] \
            or emit.shape[0] != trans.shape[0]:
        raise ValueError("invalid HMM array shapes")
    if np.any(start < 0.0) or np.any(trans < 0.0) or np.any(emit < 0.0):
        raise ValueError("HMM probabilities must be non-negative")
    if np.any(sequence < 0) or np.any(sequence >= emit.shape[1]):
        raise ValueError("observation index out of range")
    if not np.isfinite(start).all():
        raise ValueError("start must contain finite values")
    return (
        np.ascontiguousarray(sequence),
        np.ascontiguousarray(start),
        np.ascontiguousarray(trans),
        np.ascontiguousarray(emit),
    )


def hmm_forward(sequence, start, trans, emit):
    sequence, start, trans, emit = _hmm_inputs(sequence, start, trans, emit)
    cpp = _load_cpp_ml_kernels()
    cpp_hmm_forward = getattr(cpp, "hmm_forward", None) if cpp is not None else None
    if callable(cpp_hmm_forward):
        return cpp_hmm_forward(sequence, start, trans, emit)
    log_start = np.log(start + 1e-12)
    log_trans = np.log(trans + 1e-12)
    log_emit = np.log(emit + 1e-12)
    output = np.empty((sequence.size, trans.shape[0]))
    output[0] = log_start + log_emit[:, sequence[0]]
    for step in range(1, sequence.size):
        output[step] = log_emit[:, sequence[step]] + np.logaddexp.reduce(
            output[step - 1][:, None] + log_trans, axis=0
        )
    return output


def hmm_backward(sequence, start, trans, emit):
    sequence, start, trans, emit = _hmm_inputs(sequence, start, trans, emit)
    cpp = _load_cpp_ml_kernels()
    cpp_hmm_backward = getattr(cpp, "hmm_backward", None) if cpp is not None else None
    if callable(cpp_hmm_backward):
        return cpp_hmm_backward(sequence, start, trans, emit)
    log_trans = np.log(trans + 1e-12)
    log_emit = np.log(emit + 1e-12)
    output = np.empty((sequence.size, trans.shape[0]))
    output[-1] = 0.0
    for step in range(sequence.size - 2, -1, -1):
        output[step] = np.logaddexp.reduce(
            log_trans + log_emit[:, sequence[step + 1]][None, :] + output[step + 1][None, :],
            axis=1,
        )
    return output


def hmm_viterbi(sequence, start, trans, emit):
    sequence, start, trans, emit = _hmm_inputs(sequence, start, trans, emit)
    cpp = _load_cpp_ml_kernels()
    cpp_hmm_viterbi = getattr(cpp, "hmm_viterbi", None) if cpp is not None else None
    if callable(cpp_hmm_viterbi):
        return cpp_hmm_viterbi(sequence, start, trans, emit)
    log_start = np.log(start + 1e-12)
    log_trans = np.log(trans + 1e-12)
    log_emit = np.log(emit + 1e-12)
    delta = np.empty((sequence.size, trans.shape[0]))
    psi = np.empty((sequence.size, trans.shape[0]), dtype=np.int64)
    delta[0] = log_start + log_emit[:, sequence[0]]
    psi[0] = 0
    for step in range(1, sequence.size):
        scores = delta[step - 1][:, None] + log_trans
        psi[step] = np.argmax(scores, axis=0)
        delta[step] = np.max(scores, axis=0) + log_emit[:, sequence[step]]
    path = np.empty(sequence.size, dtype=np.int64)
    path[-1] = np.argmax(delta[-1])
    for step in range(sequence.size - 2, -1, -1):
        path[step] = psi[step + 1, path[step + 1]]
    return path


def radius_neighbors(X, X_fit, radius):
    """Return Euclidean radius neighbors as CSR arrays with distances."""
    X = _matrix(X, "X")
    X_fit = _matrix(X_fit, "X_fit")
    if X.shape[1] != X_fit.shape[1]:
        raise ValueError("X and X_fit must have the same number of features")
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    cpp = _load_cpp_ml_kernels()
    cpp_radius_neighbors = getattr(cpp, "radius_neighbors", None) if cpp is not None else None
    if callable(cpp_radius_neighbors):
        return cpp_radius_neighbors(X, X_fit, radius)
    distances = np.sqrt(
        np.maximum(
            np.sum(X * X, axis=1)[:, None]
            + np.sum(X_fit * X_fit, axis=1)[None, :]
            - 2.0 * (X @ X_fit.T),
            0.0,
        )
    )
    rows, cols = np.nonzero(distances <= radius)
    n = X.shape[0]
    indptr = np.bincount(rows, minlength=n).cumsum()
    indptr = np.concatenate(([0], indptr)).astype(np.int64, copy=False)
    return (
        indptr,
        cols.astype(np.int64, copy=False),
        distances[rows, cols].astype(float, copy=False),
    )


def csr_propagate(indptr, indices, weights, F):
    """Multiply a CSR graph by a dense label-distribution matrix."""
    indptr = np.asarray(indptr, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    weights = np.asarray(weights, dtype=float)
    F = _matrix(F, "F")
    n = F.shape[0]
    if indptr.ndim != 1 or indptr.shape != (n + 1,):
        raise ValueError("indptr must have n_samples + 1 entries")
    if indices.ndim != 1 or weights.ndim != 1 or indices.shape != weights.shape:
        raise ValueError("indices and weights must be matching 1D arrays")
    if indptr[0] != 0 or indptr[-1] != indices.size or np.any(np.diff(indptr) < 0):
        raise ValueError("invalid CSR row pointer")
    if np.any(indices < 0) or np.any(indices >= n) or not np.isfinite(weights).all():
        raise ValueError("invalid CSR indices or weights")
    indptr = np.ascontiguousarray(indptr)
    indices = np.ascontiguousarray(indices)
    weights = np.ascontiguousarray(weights)
    cpp = _load_cpp_ml_kernels()
    if cpp is not None:
        return cpp.csr_propagate(indptr, indices, weights, F)
    out = np.zeros_like(F, dtype=float)
    for row in range(n):
        start, end = indptr[row], indptr[row + 1]
        if start != end:
            out[row] = np.sum(weights[start:end, None] * F[indices[start:end]], axis=0)
    return out


def kmeans_assign(X, centers):
    X = _matrix(X, "X")
    centers = _matrix(centers, "centers")
    if X.shape[1] != centers.shape[1]:
        raise ValueError("X and centers must have the same number of features")
    cpp = _load_cpp_ml_kernels()
    if cpp is not None:
        return cpp.kmeans_assign(X, centers)
    x2 = np.sum(X * X, axis=1)[:, None]
    c2 = np.sum(centers * centers, axis=1)[None, :]
    distances = np.maximum(x2 + c2 - 2.0 * (X @ centers.T), 0.0)
    labels = np.argmin(distances, axis=1).astype(np.int64, copy=False)
    return labels, distances[np.arange(X.shape[0]), labels]


def kmeans_update(X, labels, n_clusters):
    X = _matrix(X, "X")
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or labels.shape[0] != X.shape[0]:
        raise ValueError("labels must match the number of samples")
    n_clusters = int(n_clusters)
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive")
    if np.any(labels < 0) or np.any(labels >= n_clusters):
        raise ValueError("labels must be between 0 and n_clusters - 1")
    cpp = _load_cpp_ml_kernels()
    if cpp is not None:
        return cpp.kmeans_update(X, labels, n_clusters)
    sums = np.zeros((n_clusters, X.shape[1]), dtype=float)
    np.add.at(sums, labels, X)
    counts = np.bincount(labels, minlength=n_clusters).astype(np.int64)
    return sums, counts


def gmm_log_gaussian(X, means, precisions, log_determinants):
    X = _matrix(X, "X")
    means = _matrix(means, "means")
    precisions = np.asarray(precisions, dtype=float)
    log_determinants = np.asarray(log_determinants, dtype=float)
    if precisions.ndim != 3 or precisions.shape != (
        means.shape[0], means.shape[1], means.shape[1]
    ):
        raise ValueError("precisions must have shape (n_components, n_features, n_features)")
    if log_determinants.shape != (means.shape[0],):
        raise ValueError("log_determinants must match the number of components")
    if X.shape[1] != means.shape[1]:
        raise ValueError("X and means must have the same number of features")
    if not np.isfinite(precisions).all() or not np.isfinite(log_determinants).all():
        raise ValueError("precisions and log_determinants must contain finite values")
    precisions = np.ascontiguousarray(precisions)
    log_determinants = np.ascontiguousarray(log_determinants)
    cpp = _load_cpp_ml_kernels()
    if cpp is not None:
        return cpp.gmm_log_gaussian(X, means, precisions, log_determinants)
    diff = X[:, None, :] - means[None, :, :]
    quad = np.einsum("nkd,kde,nke->nk", diff, precisions, diff)
    d = X.shape[1]
    return -0.5 * (d * np.log(2.0 * np.pi) + log_determinants[None, :] + quad)


def gmm_m_step(X, responsibilities, reg_covar):
    X = _matrix(X, "X")
    responsibilities = _matrix(responsibilities, "responsibilities")
    if responsibilities.shape[0] != X.shape[0]:
        raise ValueError("responsibilities must match the number of samples")
    reg_covar = float(reg_covar)
    if not np.isfinite(reg_covar) or reg_covar < 0.0:
        raise ValueError("reg_covar must be non-negative and finite")
    if np.any(responsibilities < 0.0):
        raise ValueError("responsibilities must be finite and non-negative")
    cpp = _load_cpp_ml_kernels()
    if cpp is not None:
        return cpp.gmm_m_step(X, responsibilities, reg_covar)
    n, d = X.shape
    nk = responsibilities.sum(axis=0) + 1e-12
    weights = nk / n
    means = (responsibilities.T @ X) / nk[:, None]
    diff = X[:, None, :] - means[None, :, :]
    covariances = np.einsum("nk,nkd,nke->kde", responsibilities, diff, diff) / nk[:, None, None]
    covariances += reg_covar * np.eye(d)[None, :, :]
    return weights, means, covariances
