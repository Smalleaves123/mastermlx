from __future__ import annotations

import numpy as np


def _majority(preds, classes):
    out: list[object] = []
    for col in preds.T:
        vals, cnt = np.unique(col, return_counts=True)
        out.append(vals[np.argmax(cnt)])
    result = np.asarray(out)
    return result
