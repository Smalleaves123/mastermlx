from __future__ import annotations

import numpy as np


GRAD_NAMES = ("dW_", "db_", "dgamma_", "dbeta_", "dU_", "du_", "dc_")


def clip_gradients(layers, max_norm):
    max_norm = float(max_norm)
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive")

    grads = []
    for layer in layers:
        for name in ("dW_", "db_", "dgamma_", "dbeta_"):
            value = getattr(layer, name, None)
            if value is not None:
                grads.append(value)

    if not grads:
        return 0.0, 1.0

    total_norm = float(np.sqrt(sum(np.sum(np.asarray(grad, dtype=float) ** 2) for grad in grads)))
    if total_norm <= max_norm:
        return total_norm, 1.0

    scale = max_norm / (total_norm + 1e-12)
    for layer in layers:
        for name in ("dW_", "db_", "dgamma_", "dbeta_"):
            value = getattr(layer, name, None)
            if value is not None:
                setattr(layer, name, value * scale)
    return total_norm, scale


def accumulate_gradients(layers, store):
    """Add current layer gradients to an accumulation mapping."""
    for layer_idx, layer in enumerate(layers):
        for name in GRAD_NAMES:
            value = getattr(layer, name, None)
            if value is None:
                continue
            key = (layer_idx, name)
            value = np.asarray(value, dtype=float)
            if key in store:
                store[key] += value
            else:
                store[key] = value.copy()


def load_accumulated_gradients(layers, store, count):
    """Replace current gradients with the averaged accumulated gradients."""
    count = int(count)
    if count < 1:
        raise ValueError("count must be positive")
    for layer_idx, layer in enumerate(layers):
        for name in GRAD_NAMES:
            key = (layer_idx, name)
            if key in store:
                setattr(layer, name, store[key] / count)


clip_grads = clip_gradients
