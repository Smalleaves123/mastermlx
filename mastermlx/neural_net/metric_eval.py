"""Metric evaluation helpers for supervised neural-network training."""

from __future__ import annotations

import numpy as np

from ..utils.metrics import (
    accuracy,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


def _softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _sigmoid(logits):
    logits = np.asarray(logits, dtype=float)
    out = np.empty_like(logits)
    positive = logits >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    out[~positive] = exp_logits / (1.0 + exp_logits)
    return out


def evaluate_metrics(task, metric_names, y_true, output, classes=None):
    """Evaluate configured metrics from model outputs."""
    names = (metric_names,) if isinstance(metric_names, str) else tuple(metric_names or ())
    if not names:
        return {}
    y_true = np.asarray(y_true)
    output = np.asarray(output, dtype=float)
    result = {}

    if task == "classification":
        classes = np.asarray(classes)
        probabilities = _softmax(output)
        indices = np.argmax(probabilities, axis=1)
        prediction = classes[indices]
        average = "binary" if classes.size == 2 else "macro"
        for name in names:
            key = str(name).lower()
            if key == "accuracy":
                result[key] = accuracy(y_true, prediction)
            elif key == "precision":
                result[key] = precision_score(y_true, prediction, average=average)
            elif key == "recall":
                result[key] = recall_score(y_true, prediction, average=average)
            elif key == "f1":
                result[key] = f1_score(y_true, prediction, average=average)
            elif key == "roc_auc":
                if classes.size != 2:
                    raise ValueError("roc_auc training metric requires binary classification")
                result[key] = roc_auc_score(y_true, probabilities[:, 1])
            elif key == "log_loss":
                result[key] = log_loss(y_true, probabilities)
            else:
                raise ValueError(f"Unsupported classification metric: {name}")
        return result

    if task in {"regression", "multioutput_regression"}:
        target = np.asarray(y_true, dtype=float)
        prediction = output if output.ndim > 1 else output.reshape(-1)
        if target.ndim == 1 and prediction.ndim == 2 and prediction.shape[1] == 1:
            target = target[:, None]
        for name in names:
            key = str(name).lower()
            if key == "mae":
                result[key] = mean_absolute_error(target, prediction)
            elif key == "mse":
                result[key] = mean_squared_error(target, prediction)
            elif key == "rmse":
                result[key] = root_mean_squared_error(target, prediction)
            elif key == "r2":
                result[key] = r2_score(target, prediction)
            else:
                raise ValueError(f"Unsupported regression metric: {name}")
        return result

    if task == "multilabel":
        target = np.asarray(y_true, dtype=int)
        probabilities = _sigmoid(output)
        prediction = (probabilities >= 0.5).astype(int)
        for name in names:
            key = str(name).lower()
            if key == "accuracy":
                result[key] = float(np.mean(np.all(target == prediction, axis=1)))
            elif key in {"precision", "recall", "f1"}:
                scorer = {"precision": precision_score, "recall": recall_score, "f1": f1_score}[key]
                values = [scorer(target[:, i], prediction[:, i]) for i in range(target.shape[1])]
                result[key] = float(np.mean(values))
            elif key == "roc_auc":
                values = [roc_auc_score(target[:, i], probabilities[:, i]) for i in range(target.shape[1])]
                result[key] = float(np.mean(values))
            else:
                raise ValueError(f"Unsupported multilabel metric: {name}")
        return result

    raise ValueError(f"Unsupported task: {task}")


__all__ = ["evaluate_metrics"]
