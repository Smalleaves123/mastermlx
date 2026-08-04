import numpy as np
import pytest

from mastermlx.neighbors import KNNClassifier, KNNRegressor, RadiusNeighborsClassifier, RadiusNeighborsRegressor
from mastermlx.neighbors._base import knn_neighbors


def test_knn_classifier_predicts_single_sample():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    model = KNNClassifier(k=1).fit(X, y)
    pred = model.predict([2.2])

    assert pred == 1


def test_knn_regressor_predicts_single_sample():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    model = KNNRegressor(k=2).fit(X, y)
    pred = model.predict([1.5])

    assert np.isclose(pred, 1.5, atol=1e-8)


def test_knn_classifier_supports_distance_weights():
    X = np.array([[0.0], [2.0], [3.0]])
    y = np.array([0, 1, 1])

    model = KNNClassifier(k=3, weights="distance").fit(X, y)
    pred = model.predict([0.1])

    assert pred == 0


def test_knn_classifier_supports_string_labels():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array(["cat", "cat", "dog"], dtype=object)

    model = KNNClassifier(k=2).fit(X, y)
    pred = model.predict([0.9])

    assert pred == "cat"


def test_knn_regressor_supports_distance_weights():
    X = np.array([[0.0], [2.0], [4.0]])
    y = np.array([0.0, 2.0, 10.0])

    model = KNNRegressor(k=3, weights="distance").fit(X, y)
    pred = model.predict([0.1])

    assert pred < 2.0


def test_knn_distance_weights_use_query_neighbor_positions():
    X = np.arange(10, dtype=float)[:, None]
    query = [[4.5]]

    classifier = KNNClassifier(k=3, weights="distance").fit(
        X, np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0])
    )
    assert classifier.predict(query) == 1

    values = np.arange(10, dtype=float)
    regressor = KNNRegressor(k=3, weights="distance").fit(X, values)
    expected = np.average([4.0, 5.0, 3.0], weights=[2.0, 2.0, 2.0 / 3.0])
    assert np.isclose(regressor.predict(query), expected)


def test_knn_neighbors_returns_exact_top_k_for_large_training_set():
    rng = np.random.default_rng(12)
    X_train = rng.normal(size=(128, 4))
    X_query = rng.normal(size=(5, 4))
    k = 7

    indices, distances = knn_neighbors(X_query, X_train, k, "euclidean")
    all_distances = np.sqrt(np.sum((X_query[:, None, :] - X_train[None, :, :]) ** 2, axis=2))
    expected = np.argsort(all_distances, axis=1, kind="stable")[:, :k]

    assert indices.shape == (X_query.shape[0], k)
    assert distances.shape == (X_query.shape[0], k)
    assert np.array_equal(np.sort(indices, axis=1), np.sort(expected, axis=1))
    expected_distances = np.take_along_axis(all_distances, expected, axis=1)
    expected_distances.sort(axis=1)
    assert np.allclose(np.sort(distances, axis=1), expected_distances)


def test_knn_neighbors_uses_stable_index_ties_for_numpy_distance_metrics():
    indices, distances = knn_neighbors(
        np.zeros((1, 2)), np.zeros((6, 2)), 2, "manhattan"
    )

    assert np.array_equal(indices, [[0, 1]])
    assert np.array_equal(distances, [[0.0, 0.0]])


def test_radius_neighbors_classifier_predicts_within_radius():
    X = np.array([[0.0], [0.3], [1.5], [1.7]])
    y = np.array([0, 0, 1, 1])

    model = RadiusNeighborsClassifier(radius=0.4).fit(X, y)
    pred = model.predict([1.6])

    assert pred == 1


def test_radius_neighbors_regressor_predicts_within_radius():
    X = np.array([[0.0], [0.2], [1.0], [1.2]])
    y = np.array([0.0, 0.2, 1.0, 1.2])

    model = RadiusNeighborsRegressor(radius=0.25).fit(X, y)
    pred = model.predict([0.1])

    assert np.isclose(pred, 0.1, atol=1e-8)


def test_radius_neighbors_uses_nearest_fallback_when_radius_is_empty():
    X = np.array([[0.0], [10.0]])
    y_classifier = np.array([0, 1])
    y_regressor = np.array([2.0, 8.0])

    assert RadiusNeighborsClassifier(radius=0.1).fit(X, y_classifier).predict([4.0]) == 0
    assert np.isclose(RadiusNeighborsRegressor(radius=0.1).fit(X, y_regressor).predict([9.0]), 8.0)


@pytest.mark.parametrize("estimator", [RadiusNeighborsClassifier, RadiusNeighborsRegressor])
def test_radius_neighbors_rejects_nonfinite_radius(estimator):
    with pytest.raises(ValueError, match="positive and finite"):
        estimator(radius=np.inf).fit(np.array([[0.0]]), np.array([0.0]))
