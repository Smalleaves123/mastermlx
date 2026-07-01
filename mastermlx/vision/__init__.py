"""Vision utilities for image preprocessing and lightweight detection workflows."""

from .core import (
    box_blur,
    integral_image,
    normalize_image,
    non_max_suppression,
    resize_nearest,
    rgb_to_gray,
    sobel_edges,
)

__all__ = [
    "box_blur",
    "integral_image",
    "normalize_image",
    "non_max_suppression",
    "resize_nearest",
    "rgb_to_gray",
    "sobel_edges",
]
