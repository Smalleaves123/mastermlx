from __future__ import annotations

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

    # Snapshot original params if model supports it
    import copy
    try:
        orig = copy.deepcopy({k: v.copy() if hasattr(v, 'copy') else copy.deepcopy(v)
                              for k, v in vars(model).items()
                              if not k.startswith('_') and k not in ('lr', 'optimizer')})
    except Exception:
        orig = None

    for _ in range(n_iters):
        idx = rng.integers(0, n, size=bs)
        xb, yb = X[idx], y[idx]

        # Set lr on model (support common param names)
        for attr in ('lr', 'learning_rate', 'eta0'):
            if hasattr(model, attr):
                setattr(model, attr, lr)

        try:
            model.partial_fit(xb, yb)
        except AttributeError:
            model.fit(xb, yb)

        # Record loss
        if hasattr(model, 'loss_') and model.loss_:
            loss = float(model.loss_[-1])
        elif hasattr(model, 'loss_curve_') and model.loss_curve_:
            loss = float(model.loss_curve_[-1])
        else:
            try:
                loss = float(model.score(xb, yb))
            except Exception:
                loss = float(lr)

        lrs.append(lr)
        losses.append(loss)
        lr *= factor

        # Stop if loss explodes
        if len(losses) > 5 and loss > 10.0 * np.min(losses[:-1]):
            break

    # Restore original params
    if orig is not None:
        for k, v in orig.items():
            try:
                setattr(model, k, v)
            except Exception:
                pass

    return np.array(lrs, dtype=float), np.array(losses, dtype=float)
