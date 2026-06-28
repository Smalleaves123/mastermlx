import numpy as np

from mastermlx.math_tools import (
    conditional_entropy,
    cross_entropy,
    entropy,
    entropy_from_counts,
    empirical_distribution,
    joint_entropy,
    js_divergence,
    kl_divergence,
    normalized_mutual_information,
    mutual_information,
    variation_of_information,
)


def test_information_entropy_family_matches_basic_identities():
    p = np.array([0.25, 0.75])
    q = np.array([0.5, 0.5])

    h_p = entropy(p)
    ce = cross_entropy(p, q)
    kl = kl_divergence(p, q)
    js = js_divergence(p, q)

    assert np.isclose(ce, h_p + kl)
    assert kl >= 0.0
    assert js >= 0.0


def test_information_joint_and_conditional_entropy_from_samples():
    x = np.array([0, 0, 1, 1, 1, 0])
    y = np.array([0, 0, 1, 1, 0, 0])

    h_xy = joint_entropy(x, y)
    h_y = entropy(empirical_distribution(y)[1])
    h_x_given_y = conditional_entropy(x, y)
    mi = mutual_information(x, y)

    assert h_xy >= 0.0
    assert h_x_given_y >= 0.0
    assert np.isclose(mi, entropy(empirical_distribution(x)[1]) + h_y - h_xy)


def test_information_extended_helpers():
    x = np.array([0, 0, 1, 1, 1, 0])
    y = np.array([0, 0, 1, 1, 0, 0])
    counts = np.array([2, 6, 2])

    assert np.isclose(entropy_from_counts(counts), entropy(counts))
    assert 0.0 <= normalized_mutual_information(x, y) <= 1.0
    assert np.isclose(variation_of_information(x, y), entropy(empirical_distribution(x)[1]) + entropy(empirical_distribution(y)[1]) - 2 * mutual_information(x, y))
