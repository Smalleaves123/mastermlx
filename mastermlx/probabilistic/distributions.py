from __future__ import annotations

import math

import numpy as np

from .exponential_family import ExponentialFamily
from ..variational.utils import digamma, log_gamma


class GaussianDistribution(ExponentialFamily):
    """Univariate Gaussian distribution."""

    def __init__(self, mean=0.0, variance=1.0):
        self.mean = float(mean)
        self.variance = float(variance)
        if self.variance <= 0.0:
            raise ValueError("variance must be positive")

    def sufficient_statistics(self, x):
        x = np.asarray(x, dtype=float)
        return (x, x**2)

    def natural_parameters(self):
        return (self.mean / self.variance, -0.5 / self.variance)

    def log_base_measure(self, x):
        x = np.asarray(x, dtype=float)
        return np.zeros_like(x, dtype=float)

    def log_normalizer(self):
        return 0.5 * (self.mean**2 / self.variance + np.log(2.0 * np.pi * self.variance))

    def sample(self, n_samples=1, random_state=None):
        rng = np.random.default_rng(random_state)
        samples = rng.normal(loc=self.mean, scale=np.sqrt(self.variance), size=int(n_samples))
        return float(samples[0]) if int(n_samples) == 1 else samples


class GammaDistribution(ExponentialFamily):
    """Gamma distribution parameterized by shape and rate."""

    def __init__(self, shape=1.0, rate=1.0):
        self.shape = float(shape)
        self.rate = float(rate)
        if self.shape <= 0.0 or self.rate <= 0.0:
            raise ValueError("shape and rate must be positive")

    def sufficient_statistics(self, x):
        x = np.asarray(x, dtype=float)
        if np.any(x <= 0.0):
            raise ValueError("x must be positive")
        return (np.log(x), x)

    def natural_parameters(self):
        return (self.shape - 1.0, -self.rate)

    def log_base_measure(self, x):
        x = np.asarray(x, dtype=float)
        return np.zeros_like(x, dtype=float)

    def log_normalizer(self):
        return math.lgamma(self.shape) - self.shape * np.log(self.rate)

    def mean(self):
        return self.shape / self.rate

    def variance(self):
        return self.shape / (self.rate**2)

    def expected_log(self):
        return float(digamma(np.array(self.shape)) - np.log(self.rate))

    def sample(self, n_samples=1, random_state=None):
        rng = np.random.default_rng(random_state)
        samples = rng.gamma(shape=self.shape, scale=1.0 / self.rate, size=int(n_samples))
        return float(samples[0]) if int(n_samples) == 1 else samples


class BetaDistribution(ExponentialFamily):
    """Beta distribution on the unit interval."""

    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = float(alpha)
        self.beta = float(beta)
        if self.alpha <= 0.0 or self.beta <= 0.0:
            raise ValueError("alpha and beta must be positive")

    def sufficient_statistics(self, x):
        x = np.asarray(x, dtype=float)
        if np.any((x <= 0.0) | (x >= 1.0)):
            raise ValueError("x must be in (0, 1)")
        return (np.log(x), np.log(1.0 - x))

    def natural_parameters(self):
        return (self.alpha - 1.0, self.beta - 1.0)

    def log_base_measure(self, x):
        x = np.asarray(x, dtype=float)
        return np.zeros_like(x, dtype=float)

    def log_normalizer(self):
        return math.lgamma(self.alpha) + math.lgamma(self.beta) - math.lgamma(self.alpha + self.beta)

    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    def variance(self):
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total**2 * (total + 1.0))

    def expected_log(self):
        total = self.alpha + self.beta
        return (
            float(digamma(np.array(self.alpha)) - digamma(np.array(total))),
            float(digamma(np.array(self.beta)) - digamma(np.array(total))),
        )

    def sample(self, n_samples=1, random_state=None):
        rng = np.random.default_rng(random_state)
        samples = rng.beta(a=self.alpha, b=self.beta, size=int(n_samples))
        return float(samples[0]) if int(n_samples) == 1 else samples


class DirichletDistribution(ExponentialFamily):
    """Dirichlet distribution over probability vectors."""

    def __init__(self, alpha):
        self.alpha = np.asarray(alpha, dtype=float)
        if self.alpha.ndim != 1 or self.alpha.size == 0:
            raise ValueError("alpha must be a non-empty 1D array")
        if np.any(self.alpha <= 0.0):
            raise ValueError("alpha must be positive")

    def sufficient_statistics(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.alpha.shape[0]:
            raise ValueError("x has a different dimension than alpha")
        if np.any(x <= 0.0):
            raise ValueError("x must be strictly positive")
        if not np.allclose(np.sum(x, axis=1), 1.0, atol=1e-8):
            raise ValueError("rows of x must sum to 1")
        return (np.log(x),)

    def natural_parameters(self):
        return (self.alpha - 1.0,)

    def log_base_measure(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return np.array(0.0)
        return np.zeros(x.shape[0], dtype=float)

    def log_normalizer(self):
        return float(np.sum(log_gamma(self.alpha)) - math.lgamma(float(np.sum(self.alpha))))

    def mean(self):
        return self.alpha / np.sum(self.alpha)

    def expected_log(self):
        total = digamma(np.array(np.sum(self.alpha)))
        return digamma(self.alpha) - total

    def sample(self, n_samples=1, random_state=None):
        rng = np.random.default_rng(random_state)
        samples = rng.dirichlet(self.alpha, size=int(n_samples))
        return samples[0] if int(n_samples) == 1 else samples
