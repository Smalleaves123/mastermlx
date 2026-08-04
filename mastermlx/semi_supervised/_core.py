from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..accel.ml_kernels import csr_propagate as _csr_propagate
from ..accel.ml_kernels import knn_graph as _accelerated_knn_graph
from ..accel.ml_kernels import knn_affinity as _accelerated_knn_affinity
from ..accel.ml_kernels import rbf_affinity as _accelerated_rbf_affinity
from ..utils import check_1d_array, check_2d_array


def pairwise_sqdist(X, Y=None):
    X = check_2d_array(X).astype(float)
    if Y is None:
        Y = X
    else:
        Y = check_2d_array(Y).astype(float)
    x2 = np.sum(X ** 2, axis=1, keepdims=True)
    y2 = np.sum(Y ** 2, axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2.0 * (X @ Y.T), 0.0)


def rbf_affinity(X, gamma):
    return _accelerated_rbf_affinity(X, gamma)


def knn_affinity(X, n_neighbors):
    return _accelerated_knn_affinity(X, n_neighbors)


@dataclass
class _SparseAffinity:
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray

    def _row_sums(self):
        rows = np.repeat(np.arange(self.indptr.size - 1), np.diff(self.indptr))
        return np.bincount(rows, weights=self.data, minlength=self.indptr.size - 1)

    def row_normalized(self):
        sums = self._row_sums()
        sums = np.where(sums == 0.0, 1.0, sums)
        rows = np.repeat(np.arange(sums.size), np.diff(self.indptr))
        return _SparseAffinity(self.indptr, self.indices, self.data / sums[rows])

    def sym_normalized(self):
        sums = self._row_sums()
        sums = np.where(sums == 0.0, 1.0, sums)
        rows = np.repeat(np.arange(sums.size), np.diff(self.indptr))
        values = self.data / np.sqrt(sums[rows] * sums[self.indices])
        return _SparseAffinity(self.indptr, self.indices, values)

    def propagate(self, F):
        return _csr_propagate(self.indptr, self.indices, self.data, F)


def knn_sparse_affinity(X, n_neighbors):
    return _SparseAffinity(*_accelerated_knn_graph(X, n_neighbors))


def row_norm(A):
    A = np.asarray(A, dtype=float)
    s = np.sum(A, axis=1, keepdims=True)
    s = np.where(s == 0.0, 1.0, s)
    return A / s


def sym_norm(A):
    A = np.asarray(A, dtype=float)
    d = np.sum(A, axis=1)
    d = np.where(d == 0.0, 1.0, d)
    s = 1.0 / np.sqrt(d)
    return (s[:, None] * A) * s[None, :]


def make_y(y):
    y = check_1d_array(y, name="y")
    classes = np.unique(y[y != -1])
    if classes.size == 0:
        raise ValueError("y must contain at least one labeled sample")
    return y, classes


def one_hot(y, classes):
    y = np.asarray(y)
    out = np.zeros((y.shape[0], classes.shape[0]), dtype=float)
    for i, cls in enumerate(classes):
        out[y == cls, i] = 1.0
    return out


def hard_labels(F, classes):
    return classes[np.argmax(F, axis=1)]
