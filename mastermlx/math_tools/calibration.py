from __future__ import annotations

import numpy as np


def _to_positive_probs(y_prob, pos_label=None):
    """Extract probabilities for the positive class.

    If y_prob is 1D, returns y_prob directly (assumed to be p(y=1)).
    If y_prob is 2D, returns the column for `pos_label` (default: last column).
    """
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim == 1:
        return np.clip(y_prob, 0.0, 1.0)
    if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
        if pos_label is None:
            return np.clip(y_prob[:, -1], 0.0, 1.0)
        if isinstance(pos_label, int) and 0 <= pos_label < y_prob.shape[1]:
            return np.clip(y_prob[:, pos_label], 0.0, 1.0)
        raise ValueError(f"pos_label={pos_label} is not a valid column index for y_prob with {y_prob.shape[1]} columns")
    raise ValueError("y_prob must be a 1D probability vector or a 2D class-probability matrix")


def _binarize_labels(y_true, pos_label=None):
    """Convert categorical labels to {0,1}, using pos_label as the positive class."""
    y_true = np.asarray(y_true)
    uniq = np.unique(y_true)
    if uniq.size == 2:
        if pos_label is None:
            pos = uniq[-1]
        else:
            pos = pos_label
        return (y_true == pos).astype(float)
    if pos_label is None:
        raise ValueError("multiclass y_true requires pos_label to specify the positive class")
    return (y_true == pos_label).astype(float)


def brier_score(y_true, y_prob, pos_label=None):
    """Brier score (mean squared error of predicted probabilities)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.ndim != 1 or y_true.size == 0:
        raise ValueError("y_true must be a non-empty 1D array")

    if y_prob.ndim == 2:
        if y_prob.shape[0] != y_true.shape[0]:
            raise ValueError("y_true and y_prob must have the same length")
        # Multiclass: one-hot encode y_true from sorted(unique)
        classes = np.unique(y_true)
        if y_prob.shape[1] != classes.shape[0]:
            raise ValueError("y_prob column count must match the number of classes in y_true")
        idx = np.searchsorted(classes, y_true)
        one_hot = np.zeros_like(y_prob, dtype=float)
        one_hot[np.arange(y_true.shape[0]), idx] = 1.0
        return float(np.mean(np.sum((one_hot - y_prob) ** 2, axis=1)))

    # Binary 1D case
    prob = _to_positive_probs(y_prob, pos_label=pos_label)
    if prob.shape[0] != y_true.shape[0]:
        raise ValueError("y_true and y_prob must have the same length")
    y_bin = _binarize_labels(y_true, pos_label=pos_label)
    return float(np.mean((y_bin - prob) ** 2))


def reliability_curve(y_true, y_prob, n_bins=10, pos_label=None):
    """Compute reliability curve: bin predicted probabilities, plot vs observed frequency."""
    y_true = np.asarray(y_true)
    y_bin = _binarize_labels(y_true, pos_label=pos_label)
    prob = _to_positive_probs(y_prob, pos_label=pos_label)

    if y_true.ndim != 1 or y_true.shape[0] != prob.shape[0]:
        raise ValueError("y_true and y_prob must have the same length")
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(prob, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    centers = 0.5 * (bins[:-1] + bins[1:])
    prob_true = np.zeros(n_bins, dtype=float)
    prob_pred = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = bin_ids == i
        counts[i] = int(np.sum(mask))
        if counts[i] > 0:
            prob_true[i] = float(np.mean(y_bin[mask]))
            prob_pred[i] = float(np.mean(prob[mask]))
        else:
            prob_true[i] = np.nan
            prob_pred[i] = np.nan
    return centers, prob_true, prob_pred, counts


def expected_calibration_error(y_true, y_prob, n_bins=10, pos_label=None):
    centers, prob_true, prob_pred, counts = reliability_curve(y_true, y_prob, n_bins=n_bins, pos_label=pos_label)
    total = np.sum(counts)
    if total == 0:
        return 0.0
    mask = counts > 0
    return float(np.sum((counts[mask] / total) * np.abs(prob_true[mask] - prob_pred[mask])))


def maximum_calibration_error(y_true, y_prob, n_bins=10, pos_label=None):
    _, prob_true, prob_pred, counts = reliability_curve(y_true, y_prob, n_bins=n_bins, pos_label=pos_label)
    mask = counts > 0
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(prob_true[mask] - prob_pred[mask])))


__all__ = [
    "brier_score",
    "expected_calibration_error",
    "maximum_calibration_error",
    "reliability_curve",
]
