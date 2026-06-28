from __future__ import annotations

import numpy as np

from ..utils import as_2d, check_1d_array, check_2d_array, clone


def _clone_list(models):
    return [clone(model) for model in models]


def _majority(preds, classes):
    out = []
    for col in preds.T:
        vals, cnt = np.unique(col, return_counts=True)
        out.append(vals[np.argmax(cnt)])
    out = np.asarray(out)
    return out[0] if out.shape[0] == 1 else out


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

