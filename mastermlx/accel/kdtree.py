"""KD-Tree wrapper with NumPy fallback."""
from __future__ import annotations
import numpy as np

try:
    from ._kdtree import kdtree_new, kdtree_query
    _HAS_KDTREE = True
except ImportError:
    _HAS_KDTREE = False


def knn_search(X_train, X_query, k):
    """Find k nearest neighbors of X_query in X_train.
    Returns (indices, distances). Falls back to brute force if KD-Tree unavailable.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_query = np.asarray(X_query, dtype=np.float64)
    if X_train.ndim != 2 or X_query.ndim != 2:
        raise ValueError("X_train and X_query must be 2D")
    if X_train.shape[1] != X_query.shape[1]:
        raise ValueError("Feature dimensions must match")
    k = int(k)
    if k < 1 or k > X_train.shape[0]:
        raise ValueError(f"k must be in [1, {X_train.shape[0]}]")

    if _HAS_KDTREE and X_train.shape[0] > 100:
        tree = kdtree_new(X_train)
        idx, dst_sq = kdtree_query(tree, X_query, np.array([k], dtype=np.int32))
        return idx, np.sqrt(np.maximum(dst_sq, 0.0))

    # Brute force fallback
    sq = np.sum(X_query**2, axis=1)[:, None] + np.sum(X_train**2, axis=1)[None, :] \
         - 2.0 * (X_query @ X_train.T)
    sq = np.maximum(sq, 0.0)
    idx = np.argpartition(sq, k, axis=1)[:, :k]
    row_idx = np.arange(X_query.shape[0])[:, None]
    return idx, sq[row_idx, idx]
