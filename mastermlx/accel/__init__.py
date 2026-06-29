"""Compute backends and acceleration helpers."""

from .cnn_ops import col2im, im2col, maxpool_backward, maxpool_forward
from .backends import (
    active_backend,
    get_active_backend,
    pairwise_dist,
    pairwise_distances,
    pairwise_manhattan,
    pairwise_manhattan_distances,
    pairwise_sq_euclid,
    pairwise_squared_euclidean,
)

__all__ = [
    "active_backend",
    "col2im",
    "im2col",
    "get_active_backend",
    "pairwise_dist",
    "pairwise_distances",
    "pairwise_manhattan",
    "pairwise_manhattan_distances",
    "pairwise_sq_euclid",
    "pairwise_squared_euclidean",
    "maxpool_backward",
    "maxpool_forward",
]
