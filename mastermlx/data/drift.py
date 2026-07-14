"""Lightweight train/test distribution drift checks."""

from __future__ import annotations

import numpy as np

from .quality import _as_table, _counts, _is_numeric, _key, _missing_mask
from .schema import compare_schema


def drift_report(train, test, *, bins=10):
    """Measure simple per-column distribution changes between two tables."""

    if int(bins) < 2:
        raise ValueError("bins must be at least 2")
    bins = int(bins)
    train_arr, train_names = _as_table(train)
    test_arr, test_names = _as_table(test)
    test_map = {name: idx for idx, name in enumerate(test_names)}
    common = [name for name in train_names if name in test_map]
    columns = []

    for idx, name in enumerate(train_names):
        if name not in test_map:
            continue
        test_idx = test_map[name]
        train_col = train_arr[:, idx]
        test_col = test_arr[:, test_idx]
        train_missing = _missing_mask(train_col)
        test_missing = _missing_mask(test_col)
        item = {
            "name": name,
            "train_missing_rate": float(train_missing.mean()),
            "test_missing_rate": float(test_missing.mean()),
            "missing_rate_diff": float(test_missing.mean() - train_missing.mean()),
        }
        if _is_numeric(train_col, train_missing) and _is_numeric(test_col, test_missing):
            a = np.asarray(train_col[~train_missing], dtype=float)
            b = np.asarray(test_col[~test_missing], dtype=float)
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]
            if a.size and b.size:
                edges = np.histogram_bin_edges(a, bins=bins)
                if np.all(edges == edges[0]):
                    edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
                train_hist, _ = np.histogram(a, bins=edges)
                test_hist, _ = np.histogram(b, bins=edges)
                train_prob = (train_hist + 1.0) / (train_hist.sum() + len(train_hist))
                test_prob = (test_hist + 1.0) / (test_hist.sum() + len(test_hist))
                item.update(
                    kind="numeric",
                    train_mean=float(np.mean(a)),
                    test_mean=float(np.mean(b)),
                    mean_diff=float(np.mean(b) - np.mean(a)),
                    train_std=float(np.std(a)),
                    test_std=float(np.std(b)),
                    psi=float(np.sum((test_prob - train_prob) * np.log(test_prob / train_prob))),
                )
            else:
                item.update(kind="numeric", train_mean=np.nan, test_mean=np.nan, mean_diff=np.nan, psi=np.nan)
        else:
            train_counts = dict((_key(value), count) for value, count in _counts(train_col[~train_missing]))
            test_counts = dict((_key(value), count) for value, count in _counts(test_col[~test_missing]))
            keys = set(train_counts) | set(test_counts)
            train_total = max(1, sum(train_counts.values()))
            test_total = max(1, sum(test_counts.values()))
            tvd = 0.5 * sum(
                abs(train_counts.get(key, 0) / train_total - test_counts.get(key, 0) / test_total)
                for key in keys
            )
            item.update(
                kind="categorical",
                train_unique_count=len(train_counts),
                test_unique_count=len(test_counts),
                unseen_rate=float(
                    sum(1 for value in test_col[~test_missing] if _key(value) not in train_counts)
                    / max(1, (~test_missing).sum())
                ),
                tvd=float(tvd),
            )
        columns.append(item)
    return {"schema": compare_schema(train, test), "columns": columns, "common_columns": common}


data_drift = drift_report

__all__ = ["data_drift", "drift_report"]
