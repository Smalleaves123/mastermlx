from __future__ import annotations

import copy
import numpy as np


def lr_find(model, X, y, start=1e-8, end=10.0, n_iters=100, batch_size=None, random_state=None):
    """Run a learning-rate range test (Leslie Smith 2015).

    Trains the model for `n_iters` mini-batches with exponentially
    increasing lr from `start` to `end`. Returns (lrs, losses) where
    the steepest loss drop indicates a good lr range.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    bs = int(batch_size) if batch_size else min(128, X.shape[0])
    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    factor = (float(end) / float(start)) ** (1.0 / max(n_iters, 1))
    lr = float(start)
    lrs, losses = [], []

    original_state = copy.deepcopy(vars(model))
    try:
        for _ in range(n_iters):
            idx = rng.integers(0, n, size=bs)
            xb, yb = X[idx], y[idx]

            for attr in ('lr', 'learning_rate', 'eta0'):
                if hasattr(model, attr):
                    setattr(model, attr, lr)

            if hasattr(model, "partial_fit"):
                model.partial_fit(xb, yb)
            else:
                model.fit(xb, yb)

            if hasattr(model, 'loss_') and model.loss_:
                loss = float(model.loss_[-1])
            elif hasattr(model, 'loss_curve_') and model.loss_curve_:
                loss = float(model.loss_curve_[-1])
            elif hasattr(model, "score"):
                loss = -float(model.score(xb, yb))
            else:
                raise ValueError("model must expose a loss history or score method")

            lrs.append(lr)
            losses.append(loss)
            lr *= factor

            if len(losses) > 5 and loss > 10.0 * np.min(losses[:-1]):
                break
    finally:
        vars(model).clear()
        vars(model).update(original_state)

    return np.array(lrs, dtype=float), np.array(losses, dtype=float)
