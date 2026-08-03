import numpy as np

from mastermlx.linear_models import LogisticRegression
from mastermlx.semi_supervised import SelfTrainingClassifier


def test_self_training_absorbs_confident_labels_and_predicts_new_samples():
    X = np.array([[0.0], [0.2], [0.4], [4.6], [4.8], [5.0]])
    y = np.array([0, -1, -1, 1, -1, -1])
    model = SelfTrainingClassifier(
        LogisticRegression(lr=0.2, n_iter=500, random_state=0),
        threshold=0.65,
        max_iter=5,
    ).fit(X, y)

    assert model.n_pseudo_labels_ >= 3
    assert np.array_equal(model.predict(), np.array([0, 0, 0, 1, 1, 1]))
    assert model.predict([[0.1]]).shape == (1,)
    assert model.label_distributions_.shape == (6, 2)
    assert np.all(model.labeled_iter_[[0, 3]] == 0)


def test_self_training_k_best_can_stop_when_confidence_is_insufficient():
    X = np.array([[-1.0], [1.0], [-0.1], [0.1]])
    y = np.array([0, 1, -1, -1])
    model = SelfTrainingClassifier(
        LogisticRegression(lr=0.05, n_iter=10, random_state=0),
        criterion="k_best",
        k_best=1,
        threshold=0.999999,
        max_iter=3,
    ).fit(X, y)

    assert model.n_pseudo_labels_ == 0
    assert np.all(model.labeled_iter_[2:] == -1)
