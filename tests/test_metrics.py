import numpy as np

from mastermlx.utils import (
    confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)


def test_confusion_matrix_counts_and_normalization():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    cm = confusion_matrix(y_true, y_pred)
    cm_true = confusion_matrix(y_true, y_pred, normalize="true")

    assert np.array_equal(cm, np.array([[1, 1], [0, 2]]))
    assert np.allclose(cm_true, np.array([[0.5, 0.5], [0.0, 1.0]]))


def test_confusion_matrix_preserves_explicit_label_order():
    y_true = np.array(["dog", "cat", "dog", "bird"])
    y_pred = np.array(["cat", "cat", "dog", "bird"])
    labels = np.array(["bird", "cat", "dog"])

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    assert np.array_equal(cm, np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]]))


def test_precision_recall_f1_binary():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1])

    assert np.isclose(precision_score(y_true, y_pred), 2 / 3)
    assert np.isclose(recall_score(y_true, y_pred), 2 / 3)
    assert np.isclose(f1_score(y_true, y_pred), 2 / 3)


def test_precision_recall_f1_macro():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 2])

    assert 0.0 <= precision_score(y_true, y_pred, average="macro") <= 1.0
    assert 0.0 <= recall_score(y_true, y_pred, average="macro") <= 1.0
    assert 0.0 <= f1_score(y_true, y_pred, average="macro") <= 1.0


def test_log_loss_binary_probabilities():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.8, 0.9, 0.2])

    loss = log_loss(y_true, y_prob)

    assert loss > 0.0
    assert loss < 0.5


def test_roc_auc_score_binary():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])

    assert np.isclose(roc_auc_score(y_true, y_score), 0.75)


def test_roc_auc_score_averages_tied_scores():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.full(4, 0.5)

    assert np.isclose(roc_auc_score(y_true, y_score), 0.5)


def test_roc_auc_score_has_backend_parity_for_tied_scores():
    from mastermlx import get_backend, set_backend

    y_true = np.array(["negative", "positive", "negative", "positive", "negative"])
    y_score = np.array([0.1, 0.8, 0.8, 0.8, 0.4])
    old = get_backend()
    try:
        set_backend("numpy")
        expected = roc_auc_score(y_true, y_score)
        set_backend("auto")
        actual = roc_auc_score(y_true, y_score)
    finally:
        set_backend(old)

    assert actual == expected


def test_roc_auc_score_rejects_non_finite_scores():
    with np.testing.assert_raises_regex(ValueError, "finite"):
        roc_auc_score([0, 1], [0.1, np.inf])


def test_top_k_accuracy_has_deterministic_tie_breaking_and_backend_parity():
    from mastermlx import get_backend, set_backend

    y_true = np.array(["low", "high", "middle"])
    y_score = np.full((3, 3), 0.5)
    labels = np.array(["low", "middle", "high"])
    old = get_backend()
    try:
        set_backend("numpy")
        expected = top_k_accuracy_score(y_true, y_score, k=1, labels=labels)
        set_backend("auto")
        actual = top_k_accuracy_score(y_true, y_score, k=1, labels=labels)
    finally:
        set_backend(old)

    assert expected == actual == 1.0 / 3.0


def test_top_k_accuracy_rejects_non_finite_scores_and_unknown_labels():
    with np.testing.assert_raises_regex(ValueError, "finite"):
        top_k_accuracy_score([0], [[np.nan, 0.0]], labels=[0, 1])
    with np.testing.assert_raises_regex(ValueError, "not in labels"):
        top_k_accuracy_score([2], [[0.2, 0.8]], labels=[0, 1])


def test_regression_metrics_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.5, 2.5, 4.5])

    assert np.isclose(mean_absolute_error(y_true, y_pred), 0.375)
    assert mean_absolute_percentage_error(y_true, y_pred) > 0.0
    assert explained_variance_score(y_true, y_pred) <= 1.0
    assert r2_score(y_true, y_pred) <= 1.0
