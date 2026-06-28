import numpy as np

from mastermlx.decomposition import FactorAnalysis


def test_factor_analysis_reduces_dimension():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(200, 2))
    W = np.array([[1.5, 0.3], [0.2, 1.1], [1.0, -0.4]])
    X = z @ W.T + 0.05 * rng.normal(size=(200, 3))

    model = FactorAnalysis(n_components=2)
    Z = model.fit_transform(X)

    assert Z.shape == (200, 2)
    assert model.components_.shape == (2, 3)
    assert model.noise_variance_.shape == (3,)


def test_factor_analysis_inverse_transform_shape():
    X = np.array(
        [
            [1.0, 0.0, 2.0],
            [2.0, 1.0, 1.0],
            [3.0, 1.0, 0.0],
            [4.0, 2.0, 1.0],
        ]
    )

    model = FactorAnalysis(n_components=2).fit(X)
    Z = model.transform(X)
    X2 = model.inverse_transform(Z)

    assert X2.shape == X.shape


def test_factor_analysis_noise_variance_positive():
    X = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
            [4.0, 3.0],
        ]
    )

    model = FactorAnalysis(n_components=1).fit(X)

    assert np.all(model.noise_variance_ > 0.0)
