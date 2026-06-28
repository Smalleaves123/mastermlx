import math

import numpy as np

from mastermlx.probabilistic import (
    BetaDistribution,
    DirichletDistribution,
    ExponentialFamily,
    GammaDistribution,
    GaussianDistribution,
)


def test_gaussian_distribution_log_prob_matches_closed_form():
    dist = GaussianDistribution(mean=1.5, variance=2.0)
    x = np.array([1.5, 2.5])
    logp = dist.log_prob(x)
    expected = -0.5 * np.log(2.0 * np.pi * 2.0) - ((x - 1.5) ** 2) / (2.0 * 2.0)
    assert np.allclose(logp, expected)


def test_gamma_distribution_moments_and_expected_log():
    dist = GammaDistribution(shape=3.0, rate=2.0)
    assert np.isclose(dist.mean(), 1.5)
    assert np.isclose(dist.variance(), 0.75)
    assert np.isfinite(dist.expected_log())


def test_beta_distribution_mean_and_log_prob_are_finite():
    dist = BetaDistribution(alpha=2.0, beta=3.0)
    x = np.array([0.25, 0.5])
    logp = dist.log_prob(x)
    assert np.isclose(dist.mean(), 0.4)
    assert np.all(np.isfinite(logp))


def test_dirichlet_distribution_mean_and_expected_log_shapes():
    dist = DirichletDistribution(alpha=np.array([1.0, 2.0, 3.0]))
    mean = dist.mean()
    elog = dist.expected_log()
    sample = np.array([0.2, 0.3, 0.5])
    logp = dist.log_prob(sample)

    assert mean.shape == (3,)
    assert np.isclose(np.sum(mean), 1.0)
    assert elog.shape == (3,)
    assert np.isfinite(logp)


def test_distributions_sampling_shapes_are_correct():
    gaussian = GaussianDistribution()
    gamma = GammaDistribution()
    beta = BetaDistribution()
    dirichlet = DirichletDistribution(alpha=np.array([1.0, 1.0, 1.0]))

    assert isinstance(gaussian.sample(random_state=0), float)
    assert gamma.sample(n_samples=4, random_state=0).shape == (4,)
    assert beta.sample(n_samples=5, random_state=0).shape == (5,)
    assert dirichlet.sample(n_samples=3, random_state=0).shape == (3, 3)


def test_exponential_family_base_class_is_subclassed():
    dist = GaussianDistribution()
    assert isinstance(dist, ExponentialFamily)
    assert math.isfinite(float(dist.log_normalizer()))
