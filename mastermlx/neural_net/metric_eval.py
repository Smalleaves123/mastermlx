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


def _metric_items(spec):
    if spec is None:
        return []
    if isinstance(spec, dict):
        return [(str(name), metric) for name, metric in spec.items()]
    if isinstance(spec, (tuple, list)) and len(spec) == 1 and isinstance(spec[0], dict):
        return _metric_items(spec[0])
    if isinstance(spec, str) or callable(spec):
        spec = (spec,)
    items = []
    for metric in spec:
        if isinstance(metric, tuple) and len(metric) == 2 and callable(metric[1]):
            items.append((str(metric[0]), metric[1]))
        elif callable(metric):
            items.append((getattr(metric, "__name__", metric.__class__.__name__), metric))
        else:
            items.append((str(metric), metric))
    return items


def _split_average(name):
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in {"macro", "micro", "weighted"}:
        return parts[0], parts[1]
    return name, "macro"


def _multi_score(target, prediction, kind, average):
    target = np.asarray(target, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    if target.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("multilabel targets and predictions must have the same 2D shape")

    def score(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        if kind == "precision":
            return float(tp / (tp + fp)) if tp + fp else 0.0
        if kind == "recall":
            return float(tp / (tp + fn)) if tp + fn else 0.0
        value = 2.0 * tp / (2.0 * tp + fp + fn)
        return float(value) if 2.0 * tp + fp + fn else 0.0

    if average == "micro":
        return score(target.ravel(), prediction.ravel())
    values = np.array([score(target[:, i], prediction[:, i]) for i in range(target.shape[1])])
    if average == "macro":
        return float(np.mean(values))
    if average == "weighted":
        weights = np.sum(target, axis=0).astype(float)
        return float(np.average(values, weights=weights)) if np.sum(weights) else 0.0
    raise ValueError("average must be one of: macro, micro, weighted")


def _custom_metrics(items, y_true, prediction, result):
    for name, metric in items:
        if not callable(metric):
            continue
        value = np.asarray(metric(y_true, prediction))
        if value.size != 1:
            raise ValueError(f"custom metric {name!r} must return one scalar")
        result[name] = float(value.reshape(-1)[0])


def evaluate_metrics(task, metric_names, y_true, output, classes=None):
    """Evaluate configured metrics from model outputs."""
    items = _metric_items(metric_names)
    if not items:
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
        for name, metric in items:
            key = str(name).lower()
            if callable(metric):
                continue
            if key == "accuracy":
                result[key] = accuracy(y_true, prediction)
            elif key == "precision":
                result[key] = precision_score(y_true, prediction, average=average)
            elif key == "recall":
                result[key] = recall_score(y_true, prediction, average=average)
            elif key == "f1":
                result[key] = f1_score(y_true, prediction, average=average)
            elif key == "roc_auc" or key.startswith("roc_auc_"):
                average = key.rsplit("_", 1)[1] if key != "roc_auc" else "macro"
                result[key] = roc_auc_score(y_true, probabilities, labels=classes, average=average)
            elif key == "log_loss":
                result[key] = log_loss(y_true, probabilities)
            else:
                raise ValueError(f"Unsupported classification metric: {name}")
        _custom_metrics(items, y_true, prediction, result)
        return result

    if task in {"regression", "multioutput_regression"}:
        target = np.asarray(y_true, dtype=float)
        prediction = output if output.ndim > 1 else output.reshape(-1)
        if target.ndim == 1 and prediction.ndim == 2 and prediction.shape[1] == 1:
            target = target[:, None]
        for name, metric in items:
            key = str(name).lower()
            if callable(metric):
                continue
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
        _custom_metrics(items, target, prediction, result)
        return result

    if task == "multilabel":
        target = np.asarray(y_true, dtype=int)
        probabilities = _sigmoid(output)
        prediction = (probabilities >= 0.5).astype(int)
        for name, metric in items:
            key = str(name).lower()
            if callable(metric):
                continue
            if key == "accuracy":
                result[key] = float(np.mean(np.all(target == prediction, axis=1)))
            elif key.startswith(("precision", "recall", "f1")):
                base, average = _split_average(key)
                if base not in {"precision", "recall", "f1"}:
                    raise ValueError(f"Unsupported multilabel metric: {name}")
                result[key] = _multi_score(target, prediction, base, average)
            elif key == "roc_auc" or key.startswith("roc_auc_"):
                average = key.rsplit("_", 1)[1] if key != "roc_auc" else "macro"
                values = [roc_auc_score(target[:, i], probabilities[:, i]) for i in range(target.shape[1])]
                if average == "macro":
                    result[key] = float(np.mean(values))
                elif average == "weighted":
                    result[key] = float(np.average(values, weights=np.sum(target, axis=0)))
                else:
                    raise ValueError("multilabel roc_auc supports macro or weighted")
            else:
                raise ValueError(f"Unsupported multilabel metric: {name}")
        _custom_metrics(items, target, prediction, result)
        return result

    raise ValueError(f"Unsupported task: {task}")


__all__ = ["evaluate_metrics"]
