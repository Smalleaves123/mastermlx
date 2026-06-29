import numpy as np

from mastermlx.neighbors import KNNClassifier, KNNRegressor, RadiusNeighborsClassifier, RadiusNeighborsRegressor


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
