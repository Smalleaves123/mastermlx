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


def distance_weights(dist):
    return 1.0 / np.maximum(dist, 1e-12)
