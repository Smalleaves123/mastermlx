"""Variational inference models and utilities."""

from .base import VarEst, VariationalEstimator
from .gaussian_mixture import BayesGMM, BayesianGaussianMixture, VGMM, VariationalGaussianMixture
from .linear_regression import VLinReg, VariationalLinearRegression
from .logistic_regression import VLogReg, VariationalLogisticRegression
from .poisson_regression import VPoisReg, VariationalPoissonRegression
from .utils import digamma, has_converged, log_gamma, log_sum_exp, normalize_log_probs

__all__ = [
    "BayesianGaussianMixture",
    "BayesGMM",
    "VarEst",
    "VariationalEstimator",
    "VGMM",
    "VariationalGaussianMixture",
    "VLinReg",
    "VariationalLinearRegression",
    "VLogReg",
    "VariationalLogisticRegression",
    "VPoisReg",
    "VariationalPoissonRegression",
    "digamma",
    "has_converged",
    "log_gamma",
    "log_sum_exp",
    "normalize_log_probs",
]
