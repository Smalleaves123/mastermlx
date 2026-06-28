import numpy as np

from mastermlx.svm import OneClassSVM


def test_one_class_svm_marks_far_point_as_outlier():
    X = np.array(
        [
            [0.0, 0.1],
            [0.1, 0.0],
            [-0.1, 0.0],
            [0.0, -0.1],
            [0.05, 0.03],
            [-0.04, 0.02],
        ]
    )

    model = OneClassSVM(nu=0.2, gamma=8.0, max_iter=3000).fit(X)
    pred = model.predict(X)
    outlier = model.predict([[2.0, 2.0]])

    assert np.sum(pred == 1) >= 4
    assert outlier == -1


def test_one_class_svm_returns_sample_scores():
    X = np.array([[0.0], [0.2], [0.3], [-0.1], [0.1]])
    model = OneClassSVM(nu=0.4, gamma=5.0, max_iter=2000).fit(X)

    scores = model.score_samples([[0.15], [1.5]])

    assert scores.shape == (2,)
    assert scores[0] > scores[1]
