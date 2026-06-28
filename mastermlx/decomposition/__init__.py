"""Dimensionality reduction and matrix factorization."""

from .factor_analysis import FactorAnalysis
from .ica import FastICA, ICA
from .kernel_pca import KPCA, KernelPCA
from .nmf import NMF
from .pca import PCA, PC
from .truncated_svd import TSVD, TruncatedSVD

__all__ = ["FactorAnalysis", "FastICA", "ICA", "KPCA", "KernelPCA", "NMF", "PC", "PCA", "TSVD", "TruncatedSVD"]
