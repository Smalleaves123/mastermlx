from __future__ import annotations

import numpy as np

from mastermlx.math_tools import (
    balanced_accuracy_score,
    brier_score,
    cohen_kappa_score,
    cosine_kernel,
    expected_calibration_error,
    hinge_loss,
    log_loss,
    matthews_corrcoef,
    maximum_calibration_error,
    multi_head_attention,
    precision_score,
    recall_score,
    r2_score,
    top_k_accuracy_score,
    reliability_curve,
    scaled_dot_product_attention,
    sinusoidal_positional_encoding,
    specificity_score,
    zero_one_loss,
)


def test_classification_metrics_cover_common_binary_case():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    decisions = np.array([1.5, -0.2, 0.2, 0.7])
    scores = np.array([[0.8, 0.2], [0.3, 0.7], [0.1, 0.9], [0.4, 0.6]])

    assert np.isclose(balanced_accuracy_score(y_true, y_pred), 0.75)
    assert np.isclose(cohen_kappa_score(y_true, y_pred), 0.5)
    assert np.isclose(matthews_corrcoef(y_true, y_pred), 1.0 / np.sqrt(3.0))
    assert np.isclose(hinge_loss(y_true, decisions), 1.1)
    assert np.isclose(precision_score(y_true, y_pred), 2 / 3)
    assert np.isclose(recall_score(y_true, y_pred), 1.0)
    assert np.isclose(specificity_score(y_true, y_pred), 0.5)
    assert np.isclose(zero_one_loss(y_true, y_pred), 0.25)
    assert np.isclose(top_k_accuracy_score(y_true, scores, k=1), 0.75)
    assert np.isclose(log_loss(y_true, scores), log_loss(y_true, scores))
    assert np.isclose(r2_score(np.array([1.0, 2.0]), np.array([1.0, 2.0])), 1.0)


def test_calibration_metrics_are_zero_for_perfect_predictions():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.0, 1.0, 0.0, 1.0])

    centers, prob_true, prob_pred, counts = reliability_curve(y_true, y_prob, n_bins=4)

    assert centers.shape == (4,)
    assert prob_true.shape == (4,)
    assert prob_pred.shape == (4,)
    assert counts.sum() == y_true.shape[0]
    assert np.isclose(brier_score(y_true, y_prob), 0.0)
    assert np.isclose(expected_calibration_error(y_true, y_prob, n_bins=4), 0.0)
    assert np.isclose(maximum_calibration_error(y_true, y_prob, n_bins=4), 0.0)


def test_attention_primitives_return_expected_shapes_and_values():
    query = np.array([[[1.0, 0.0]]])
    key = np.array([[[1.0, 0.0]]])
    value = np.array([[[2.0, 3.0]]])

    out, weights = scaled_dot_product_attention(query, key, value, return_weights=True)

    assert out.shape == (1, 1, 2)
    assert weights.shape == (1, 1, 1)
    assert np.allclose(out, value)
    assert np.allclose(weights, np.ones((1, 1, 1)))


def test_kernel_and_top_level_exports_are_available():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    Y = np.array([[1.0, 1.0]])

    K = cosine_kernel(X, Y)

    assert K.shape == (2, 1)


def test_multi_head_attention_and_positional_encoding_are_available():
    x = np.array([[[1.0, 0.0, 0.0, 1.0]]])
    w = np.eye(4)

    out = multi_head_attention(x, w, w, w, w, n_heads=2)
    pos = sinusoidal_positional_encoding(3, 4)

    assert out.shape == x.shape
    assert pos.shape == (3, 4)
    assert np.allclose(pos[0, 0::2], 0.0)
    assert np.allclose(pos[0, 1::2], 1.0)
