"""Probabilistic models."""

from .bayesian_regression import BayesLinReg, BayesianLinearRegression
from .distributions import BetaDistribution, DirichletDistribution, GammaDistribution, GaussianDistribution
from .discriminant_analysis import BaseDA, LDA, QDA
from .exponential_family import ExponentialFamily
from .gaussian_process import GPR, GaussianProcessRegressor
from .naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from .hmm import HMM
from .kde import KernelDensity
from ..variational.linear_regression import VLinReg, VariationalLinearRegression
from ..variational.utils import digamma, has_converged, log_gamma, log_sum_exp, normalize_log_probs

DiscriminantLDA = LDA

__all__ = [
    "BayesianLinearRegression",
    "BayesLinReg",
    "BernoulliNB",
    "BetaDistribution",
    "BaseDA",
    "DirichletDistribution",
    "DiscriminantLDA",
    "ExponentialFamily",
    "GaussianNB",
    "GammaDistribution",
    "GPR",
    "GaussianProcessRegressor",
    "GaussianDistribution",
    "HMM",
    "KernelDensity",
    "LDA",
    "MultinomialNB",
    "QDA",
    "VLinReg",
    "VariationalLinearRegression",
    "digamma",
    "has_converged",
    "log_gamma",
    "log_sum_exp",
    "normalize_log_probs",
]
