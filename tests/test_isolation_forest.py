import numpy as np

from mastermlx import IsolationForest
from mastermlx.anomaly import IsolationForest as IsolationForestPkg


def test_isolation_forest_flags_far_outlier():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.1, 0.1],
            [0.05, 0.02],
            [-0.03, -0.04],
            [0.02, -0.06],
            [3.0, 3.0],
        ]
    )

    model = IsolationForest(n_estimators=64, max_samples=1.0, contamination=0.15, random_state=0)
    model.fit(X)
    pred = model.predict(X)

    assert pred.shape == (7,)
    assert pred[-1] == -1
    assert np.sum(pred[:-1] == 1) >= 5


def test_isolation_forest_scores_outlier_higher():
    X = np.array([[0.0], [0.1], [-0.1], [0.05], [0.02], [2.5]], dtype=float)
    model = IsolationForestPkg(n_estimators=50, max_samples=1.0, contamination=0.2, random_state=1).fit(X)

    scores = model.score_samples([[0.03], [2.5]])

    assert scores.shape == (2,)
    assert scores[1] > scores[0]


def test_isolation_forest_predicts_single_sample():
    X = np.array([[0.0, 0.0], [0.1, 0.1], [-0.1, -0.1], [2.0, 2.0]], dtype=float)
    model = IsolationForest(n_estimators=32, max_samples=1.0, contamination=0.25, random_state=2).fit(X)

    pred = model.predict([2.0, 2.0])
    score = model.score_samples([2.0, 2.0])

    assert isinstance(pred, int)
    assert isinstance(score, float)
