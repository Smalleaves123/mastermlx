import numpy as np

from mastermlx.anomaly import LocalOutlierFactor


def test_lof_flags_far_point_as_outlier():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [-0.1, 0.0],
            [0.0, -0.1],
            [0.05, -0.03],
            [2.5, 2.5],
        ]
    )
    model = LocalOutlierFactor(n_neighbors=3, contamination=0.15).fit(X)

    pred = model.predict(X)
    scores = model.score_samples(X)

    assert pred[-1] == -1
    assert scores[-1] < scores[0]


def test_lof_scores_new_samples():
    X = np.array([[0.0], [0.1], [-0.1], [0.05], [0.02], [0.0]], dtype=float)
    model = LocalOutlierFactor(n_neighbors=2, contamination=0.2).fit(X)

    scores = model.score_samples([[0.01], [2.0]])
    pred = model.predict([[0.01], [2.0]])

    assert scores.shape == (2,)
    assert scores[0] > scores[1]
    assert np.array_equal(pred, np.array([1, -1]))
