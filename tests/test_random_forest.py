import numpy as np

from mastermlx.trees import RandomForestClassifier


def test_random_forest_fits_simple_data():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.2, 0.2],
        [0.9, 0.8],
    ])
    y = np.array([0, 0, 0, 1, 0, 1])

    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert model.score(X, y) >= 0.8

