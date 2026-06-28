"""Manifold learning and nonlinear embedding."""

from .mds import MDS, ClassicalMDS
from .isomap import Isomap
from .lle import LLE, LocallyLinearEmbedding
from .spectral import SpectralEmbedding

__all__ = [
    "ClassicalMDS",
    "Isomap",
    "LLE",
    "LocallyLinearEmbedding",
    "MDS",
    "SpectralEmbedding",
]
