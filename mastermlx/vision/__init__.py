"""Vision utilities for image preprocessing and lightweight detection workflows."""

from .core import (
    box_blur,
    histogram_of_oriented_gradients,
    integral_image,
    normalize_image,
    non_max_suppression,
    generate_proposals,
    patch_descriptor,
    resize_nearest,
    sliding_window,
    rgb_to_gray,
    sobel_edges,
)

__all__ = [
    "box_blur",
    "histogram_of_oriented_gradients",
    "integral_image",
    "generate_proposals",
    "normalize_image",
    "non_max_suppression",
    "patch_descriptor",
    "resize_nearest",
    "sliding_window",
    "rgb_to_gray",
    "sobel_edges",
]
