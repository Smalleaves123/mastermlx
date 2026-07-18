from __future__ import annotations

import numpy as np

from ..utils import clone


def _clone_list(models):
    return [clone(model) for model in models]


def _majority(preds, classes):
    out: list[object] = []
    for col in preds.T:
        vals, cnt = np.unique(col, return_counts=True)
        out.append(vals[np.argmax(cnt)])
    result = np.asarray(out)
    return result[0] if result.shape[0] == 1 else result


def _mean_pred(preds):
    out = np.mean(preds, axis=0)
    return float(out[0]) if out.shape[0] == 1 else out


def _softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _blend_probs(preds):
    preds = np.asarray(preds, dtype=float)
    if preds.ndim == 2:
        return np.mean(preds, axis=0)
    return np.mean(preds, axis=0)
