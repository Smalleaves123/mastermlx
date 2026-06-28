"""Compute backends and acceleration helpers."""

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
    "get_active_backend",
    "pairwise_dist",
    "pairwise_distances",
    "pairwise_manhattan",
    "pairwise_manhattan_distances",
    "pairwise_sq_euclid",
    "pairwise_squared_euclidean",
]
