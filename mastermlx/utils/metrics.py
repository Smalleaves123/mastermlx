from __future__ import annotations

import numpy as np

try:
    from ._metrics_ops import confusion_matrix_counts as _cy_confusion_matrix_counts
except ImportError:  # pragma: no cover - fallback when Cython extensions are unavailable
    _cy_confusion_matrix_counts = None


def confusion_matrix(y_true, y_pred, labels=None, normalize=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    if _cy_confusion_matrix_counts is not None:
        cm = np.asarray(_cy_confusion_matrix_counts(y_true, y_pred, labels), dtype=float)
    else:
        index = {label: idx for idx, label in enumerate(labels)}
        cm = np.zeros((labels.shape[0], labels.shape[0]), dtype=float)
        for yt, yp in zip(y_true, y_pred):
            cm[index[yt], index[yp]] += 1.0

    if normalize is None:
        return cm.astype(int)
    if normalize == "true":
        denom = cm.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        denom = cm.sum(axis=0, keepdims=True)
    elif normalize == "all":
        denom = np.array([[cm.sum()]])
    else:
        raise ValueError("normalize must be one of: None, true, pred, all")
    return np.divide(cm, denom, out=np.zeros_like(cm), where=denom != 0)


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def _precision_recall_f1(y_true, y_pred, average="binary", pos_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    labels = np.unique(np.concatenate([y_true, y_pred]))
    if average == "binary":
        if labels.shape[0] > 2:
            raise ValueError("average='binary' is only supported for binary targets")
        labels = np.array([pos_label])
    elif average not in {"macro", "micro", "weighted", None}:
        raise ValueError("average must be one of: binary, macro, micro, weighted, None")

    all_labels = np.unique(np.concatenate([y_true, y_pred]))
    full_cm = confusion_matrix(y_true, y_pred, labels=all_labels).astype(float)

    if average == "micro":
        tp = np.trace(full_cm)
        fp = np.sum(full_cm, axis=0).sum() - tp
        fn = np.sum(full_cm, axis=1).sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    tp = np.diag(full_cm)
    fp = full_cm.sum(axis=0) - tp
    fn = full_cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)

    if average is None:
        return precision, recall, f1
    if average == "binary":
        if pos_label not in all_labels:
            raise ValueError("pos_label must be present in y_true or y_pred")
        idx = int(np.flatnonzero(all_labels == pos_label)[0])
        return float(precision[idx]), float(recall[idx]), float(f1[idx])
    if average == "macro":
        return float(np.mean(precision)), float(np.mean(recall)), float(np.mean(f1))
    weights = full_cm.sum(axis=1)
    total = np.sum(weights)
    if total == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.sum(precision * weights) / total),
        float(np.sum(recall * weights) / total),
        float(np.sum(f1 * weights) / total),
    )


def precision_score(y_true, y_pred, average="binary", pos_label=1):
    return _precision_recall_f1(y_true, y_pred, average=average, pos_label=pos_label)[0]


def recall_score(y_true, y_pred, average="binary", pos_label=1):
    return _precision_recall_f1(y_true, y_pred, average=average, pos_label=pos_label)[1]


def f1_score(y_true, y_pred, average="binary", pos_label=1):
    return _precision_recall_f1(y_true, y_pred, average=average, pos_label=pos_label)[2]


def log_loss(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array")
    if y_pred.ndim == 1:
        y_pred = np.column_stack([1.0 - y_pred, y_pred])
    if y_pred.ndim != 2 or y_pred.shape[0] != y_true.shape[0]:
        raise ValueError("y_pred must be a 2D array with one row per sample")
    labels = np.unique(y_true)
    if y_pred.shape[1] != labels.shape[0]:
        raise ValueError("y_pred column count must match the number of classes")
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    y_pred = y_pred / np.sum(y_pred, axis=1, keepdims=True)
    idx = np.searchsorted(labels, y_true)
    return float(-np.mean(np.log(y_pred[np.arange(y_true.shape[0]), idx])))


def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true and y_score must be 1D arrays")
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    classes = np.unique(y_true)
    if classes.shape[0] != 2:
        raise ValueError("roc_auc_score currently supports only binary targets")

    y_bin = (y_true == classes[1]).astype(int)
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, y_score.shape[0] + 1, dtype=float)
    pos = y_bin == 1
    n_pos = np.sum(pos)
    n_neg = y_bin.shape[0] - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc_score requires both positive and negative samples")
    rank_sum = np.sum(ranks[pos])
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def specificity_score(y_true, y_pred, pos_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if classes.shape[0] != 2:
        raise ValueError("specificity_score currently supports only binary targets")
    if pos_label not in classes:
        raise ValueError("pos_label must be present in y_true or y_pred")
    neg_label = classes[0] if classes[1] == pos_label else classes[1]
    y_true_neg = y_true == neg_label
    y_pred_neg = y_pred == neg_label
    tn = np.sum(y_true_neg & y_pred_neg)
    fp = np.sum(y_true_neg & ~y_pred_neg)
    return float(tn / (tn + fp)) if tn + fp else 0.0


def zero_one_loss(y_true, y_pred, normalize=True):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    loss = np.sum(y_true != y_pred)
    return float(loss / y_true.shape[0]) if normalize else int(loss)


def top_k_accuracy_score(y_true, y_score, k=2, labels=None):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array")
    if y_score.ndim != 2 or y_score.shape[0] != y_true.shape[0]:
        raise ValueError("y_score must be a 2D array with one row per sample")
    k = int(k)
    if k < 1:
        raise ValueError("k must be at least 1")
    if k > y_score.shape[1]:
        raise ValueError("k cannot exceed the number of classes")
    # Map labels to 0-based column indices
    if labels is None:
        labels = np.unique(y_true)
    labels = np.asarray(labels)
    if labels.size != y_score.shape[1]:
        raise ValueError("labels must have the same length as y_score columns")
    # Build a dict-based index mapping (works with unsorted labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    try:
        idx = np.array([label_to_idx[yt] for yt in y_true], dtype=int)
    except KeyError as e:
        raise ValueError(f"y_true contains label {e.args[0]} not in labels") from None
    topk = np.argpartition(y_score, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(topk == idx[:, None], axis=1)))


def fbeta_score(y_true, y_pred, beta=1.0, average="binary", pos_label=1):
    p, r, _ = _precision_recall_f1(y_true, y_pred, average=average, pos_label=pos_label)
    b2 = float(beta) ** 2
    if np.ndim(p) == 0:
        denom = b2 * p + r
        return float((1.0 + b2) * p * r / denom) if denom > 0 else 0.0
    denom = b2 * p + r
    return np.divide((1.0 + b2) * p * r, denom, out=np.zeros_like(p), where=denom != 0)


def avg_precision_score(y_true, y_score):
    """Average precision (area under PR curve) for binary classification."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true and y_score must be 1D arrays")
    order = np.argsort(y_score)[::-1]
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    precision = tp / np.arange(1, len(y_true) + 1)
    recall = tp / max(np.sum(y_true), 1)
    # Compute AP as sum of precision at each recall threshold where recall changes
    mask = np.diff(recall, prepend=0) > 0
    return float(np.sum(precision[mask] * np.diff(recall, prepend=0)[mask]))


def jaccard_score(y_true, y_pred, average="binary", pos_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays")
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp_fn = np.sum(y_true != y_pred)
        return float(tp / (tp + fp_fn)) if tp + fp_fn > 0 else 0.0
    scores = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp_fn = np.sum((y_true != label) & (y_pred == label)) + np.sum((y_true == label) & (y_pred != label))
        scores.append(float(tp / (tp + fp_fn)) if tp + fp_fn > 0 else 0.0)
    if average == "macro":
        return float(np.mean(scores))
    if average == "weighted":
        weights = np.array([np.sum(y_true == l) for l in labels], dtype=float)
        return float(np.sum(weights * np.array(scores)) / np.sum(weights))
    raise ValueError("average must be 'binary', 'macro', or 'weighted'")


def mean_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-12, 1e-12, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def explained_variance_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    var_y = np.var(y_true)
    if var_y == 0.0:
        return 0.0
    return float(1.0 - np.var(diff) / var_y)


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


mae = mean_absolute_error
mse = mean_squared_error
rmse = root_mean_squared_error
mape = mean_absolute_percentage_error
expl_var = explained_variance_score
prf1 = _precision_recall_f1
