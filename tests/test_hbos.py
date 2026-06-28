import numpy as np

from mastermlx.anomaly import HBOS


def test_hbos_flags_far_outlier():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.1, 0.1],
            [0.05, 0.02],
            [-0.03, -0.04],
            [0.02, -0.06],
            [4.0, 4.0],
        ]
    )

    model = HBOS(n_bins=4, contamination=0.15).fit(X)
    pred = model.predict(X)

    assert pred.shape == (7,)
    assert pred[-1] == -1
    assert np.sum(pred[:-1] == 1) >= 5


def test_hbos_scores_outlier_higher():
    X = np.array([[0.0], [0.1], [-0.1], [0.05], [0.02], [2.5]], dtype=float)
    model = HBOS(n_bins=3, contamination=0.2).fit(X)

    scores = model.score_samples([[0.03], [2.5]])

    assert scores.shape == (2,)
    assert scores[1] > scores[0]


def test_hbos_predicts_single_sample():
    X = np.array([[0.0], [0.1], [0.2], [3.0]], dtype=float)
    model = HBOS(n_bins=3, contamination=0.25).fit(X)

    pred = model.predict([3.0])
    score = model.score_samples([3.0])

    assert isinstance(pred, int)
    assert isinstance(score, float)
