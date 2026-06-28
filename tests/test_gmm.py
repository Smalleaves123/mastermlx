import numpy as np

from mastermlx.clustering import BayesianGaussianMixture, GMM, VariationalGaussianMixture


def test_gmm_predicts_cluster_labels():
    X = np.array([
        [0.0, 0.0],
        [0.1, -0.1],
        [5.0, 5.0],
        [5.1, 4.9],
    ])

    model = GMM(n_components=2, max_iter=50, random_state=0)
    model.fit(X)
    pred = model.predict(X)

    assert pred.shape == (4,)
    assert set(pred.tolist()) <= {0, 1}


def test_gmm_responsibilities_sum_to_one():
    X = np.array([
        [0.0, 0.0],
        [5.0, 5.0],
    ])

    model = GMM(n_components=2, max_iter=10, random_state=0).fit(X)
    proba = model.predict_proba(X)

    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-8)


def test_variational_gmm_predicts_cluster_labels():
    X = np.array([
        [0.0, 0.0],
        [0.1, -0.1],
        [5.0, 5.0],
        [5.2, 4.9],
    ])

    model = VariationalGaussianMixture(n_components=2, max_iter=100, random_state=0)
    model.fit(X)
    pred = model.predict(X)

    assert pred.shape == (4,)
    assert set(pred.tolist()) <= {0, 1}


def test_variational_gmm_responsibilities_sum_to_one():
    X = np.array([
        [0.0, 0.0],
        [5.0, 5.0],
    ])

    model = VariationalGaussianMixture(n_components=2, max_iter=50, random_state=0).fit(X)
    proba = model.predict_proba(X)

    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-8)


def test_variational_gmm_shrinks_unused_component_weight():
    X = np.array([
        [-0.2, 0.0],
        [0.0, 0.1],
        [0.1, -0.1],
        [4.9, 5.1],
        [5.0, 5.0],
        [5.2, 4.8],
    ])

    model = VariationalGaussianMixture(n_components=4, max_iter=100, alpha0=0.2, random_state=0)
    model.fit(X)

    assert np.isclose(np.sum(model.weights_), 1.0)
    assert np.min(model.weights_) < 0.12
    assert len(model.lower_bound_) >= 1


def test_bayesian_gmm_tracks_active_components():
    X = np.array([
        [-0.1, 0.0],
        [0.0, 0.1],
        [0.1, -0.1],
        [5.0, 5.0],
        [5.1, 4.9],
        [4.9, 5.2],
    ])

    model = BayesianGaussianMixture(
        n_components=5,
        max_iter=100,
        weight_concentration_prior=0.2,
        random_state=0,
    ).fit(X)

    assert model.active_components_.shape == (5,)
    assert 1 <= model.n_active_components_ <= 5
    assert np.isclose(np.sum(model.weights_), 1.0)
