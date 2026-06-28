import numpy as np

from mastermlx.math_tools import (
    dropout,
    gauss,
    jitter,
    laplace,
    poisson,
    salt_pepper,
    shuffle,
    swap,
    uniform,
)


def test_gauss_preserves_shape():
    x = np.ones((10, 5))
    y = gauss(x, scale=0.1, random_state=0)
    assert y.shape == x.shape
    assert not np.allclose(y, x)


def test_gauss_deterministic():
    x = np.ones(10)
    a = gauss(x, scale=0.5, random_state=42)
    b = gauss(x, scale=0.5, random_state=42)
    assert np.array_equal(a, b)


def test_uniform_range():
    x = np.zeros(1000)
    y = uniform(x, low=-1, high=1, random_state=0)
    assert y.min() >= -1
    assert y.max() <= 1


def test_laplace_noise():
    x = np.ones(500)
    y = laplace(x, scale=0.5, random_state=0)
    assert y.shape == x.shape
    assert not np.allclose(y, x)


def test_salt_pepper():
    x = np.ones((10, 10))
    y = salt_pepper(x, prob=0.2, salt_val=99, pepper_val=-99, random_state=0)
    assert y.shape == x.shape
    assert np.any(y == 99) or np.any(y == -99)


def test_dropout_sparsity():
    x = np.ones(1000)
    y = dropout(x, prob=0.5, random_state=0)
    zeros = np.sum(y == 0)
    assert 400 < zeros < 600  # ~50%


def test_poisson_noise():
    x = np.ones(100) * 10
    y = poisson(x, random_state=0)
    assert y.shape == x.shape
    assert np.all(y >= 0)
    assert abs(np.mean(y) - 10) < 2


def test_jitter_alias():
    x = np.ones(10)
    a = jitter(x, scale=0.1, random_state=42)
    b = gauss(x, scale=0.1, random_state=42)
    assert np.array_equal(a, b)


def test_shuffle_axis():
    x = np.arange(20).reshape(4, 5)
    y = shuffle(x, axis=0, random_state=0)
    assert y.shape == x.shape
    assert not np.array_equal(y, x)


def test_swap():
    x = np.arange(10)
    y = swap(x, prob=0.5, random_state=0)
    assert y.shape == x.shape
    # after swaps, array should differ at some positions
