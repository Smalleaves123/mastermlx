from __future__ import annotations

import numpy as np

from ..math_tools.distance import pairwise_distance


def smote(X, y, k=5, random_state=None):
    """Synthetic Minority Over-sampling Technique.

    Returns balanced (X_resampled, y_resampled) by interpolating
    minority samples with their nearest neighbors.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size:
        raise ValueError("X must be 2D, y must be 1D, same length")
    uniq, counts = np.unique(y, return_counts=True)
    if uniq.size < 2:
        raise ValueError("Need at least 2 classes")
    maj_cls = uniq[np.argmax(counts)]
    maj_n = int(np.max(counts))
    rng = np.random.default_rng(random_state)

    X_res, y_res = [X], [y]
    for cls in uniq:
        if cls == maj_cls:
            continue
        mask = y == cls
        X_cls = X[mask]
        n_need = maj_n - int(np.sum(mask))
        if n_need <= 0:
            continue

        k_eff = min(k, X_cls.shape[0] - 1)
        if k_eff < 1:
            continue

        dists = pairwise_distance(X_cls, X_cls, metric="euclidean")
        # Exclude self from nearest neighbors
        np.fill_diagonal(dists, np.inf)
        nn_indices = np.argpartition(dists, k_eff, axis=1)[:, :k_eff]

        syn = []
        for _ in range(n_need):
            i = rng.integers(0, X_cls.shape[0])
            j = nn_indices[i, rng.integers(0, k_eff)]
            lam = rng.random()
            syn.append(X_cls[i] + lam * (X_cls[j] - X_cls[i]))
        X_res.append(np.array(syn, dtype=float))
        y_res.append(np.full(n_need, cls))

    return np.vstack(X_res), np.concatenate(y_res)
