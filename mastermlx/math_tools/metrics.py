from __future__ import annotations

import numpy as np

from ..utils.metrics import *  # noqa: F401,F403


def balanced_accuracy_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred).astype(float)
    row_sums = cm.sum(axis=1)
    recalls = np.divide(np.diag(cm), row_sums, out=np.zeros_like(row_sums), where=row_sums != 0)
    return float(np.mean(recalls))


def matthews_corrcoef(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred).astype(float)
    t_sum = cm.sum(axis=1)
    p_sum = cm.sum(axis=0)
    c = np.trace(cm)
    s = cm.sum()
    cov_ytyp = c * s - np.dot(t_sum, p_sum)
    cov_ypyp = s**2 - np.dot(p_sum, p_sum)
    cov_ytyt = s**2 - np.dot(t_sum, t_sum)
    denom = np.sqrt(max(cov_ytyt * cov_ypyp, 0.0))
    return float(cov_ytyp / denom) if denom > 0 else 0.0


def cohen_kappa_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred).astype(float)
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.trace(cm) / n
    pe = np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / (n * n)
    return float((po - pe) / (1.0 - pe)) if pe < 1.0 else 0.0


def hinge_loss(y_true, decision):
    y_true = np.asarray(y_true)
    decision = np.asarray(decision, dtype=float)
    if y_true.ndim != 1 or decision.ndim != 1:
        raise ValueError("y_true and decision must be 1D arrays")
    if y_true.shape[0] != decision.shape[0]:
        raise ValueError("y_true and decision must have the same length")
    y = np.where(y_true > 0, 1.0, -1.0)
    return float(np.mean(np.maximum(0.0, 1.0 - y * decision)))
