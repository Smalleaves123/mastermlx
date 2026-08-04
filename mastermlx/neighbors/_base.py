from __future__ import annotations

import numpy as np

from ..utils.distance import pairwise_distance


def check_metric(metric):
    if metric not in {"euclidean", "manhattan", "minkowski", "chebyshev", "cosine", "hamming", "jaccard", "mahalanobis"}:
        raise ValueError(
            "metric must be one of: euclidean, manhattan, minkowski, chebyshev, cosine, hamming, jaccard, mahalanobis"
        )


def check_weights(weights):
    if weights not in {"uniform", "distance"}:
        raise ValueError("weights must be 'uniform' or 'distance'")


def pairwise_neighbor_distance(X, X_fit, metric):
    # Try KD-Tree for euclidean metric on large datasets
    if metric == "euclidean" and X_fit.shape[0] > 100:
        try:
            from ..accel.kdtree import knn_search
            k = X_fit.shape[0]
            _, dists = knn_search(X_fit, X, k)
            return dists
        except ImportError:
            pass
    return pairwise_distance(X, X_fit, metric=metric)


def knn_neighbors(X, X_fit, k, metric):
    """Return the indices and distances of the ``k`` nearest samples.

    The KD-Tree path avoids materializing the full query-by-training distance
    matrix for the common Euclidean case.  Other metrics keep the existing
    pairwise-distance implementation and use a partial NumPy selection.
    """
    if metric == "euclidean" and X_fit.shape[0] > 100:
        try:
            from ..accel.kdtree import knn_search

            return knn_search(X_fit, X, k)
        except ImportError:
            pass

    dist = pairwise_distance(X, X_fit, metric=metric)
    nn = np.argpartition(dist, k - 1, axis=1)[:, :k]
    selected_dist = np.take_along_axis(dist, nn, axis=1)
    return nn, selected_dist


def distance_weights(dist):
    return 1.0 / np.maximum(dist, 1e-12)
