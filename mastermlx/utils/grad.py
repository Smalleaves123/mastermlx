from __future__ import annotations

import numpy as np


# Kept as a compatibility list for lightweight third-party layers.  Built-in
# layers are discovered dynamically from their ``d<parameter>_`` attributes.
GRAD_NAMES = (
    "dW_", "db_", "dgamma_", "dbeta_", "dU_", "du_", "dc_",
    "dW_xh_", "dW_hh_", "dW_zr_", "dW_h_", "dU_zr_", "dU_h_",
    "db_zr_", "db_h_", "dW_q_", "dW_k_", "dW_v_", "dW_o_",
    "db_q_", "db_k_", "db_v_", "db_o_",
)


def _grad_items(layer):
    """Yield ``(name, value)`` pairs for all gradients on a layer.

    Matching gradients to their parameter names keeps clipping and
    accumulation aligned as new layers add parameters, while ``GRAD_NAMES``
    preserves support for small external layer stubs without parameter attrs.
    """
    own_parameters = getattr(layer, "_own_parameters", None)
    param_names = set() if own_parameters is None else {
        name for name, _ in own_parameters()
    }
    for name, value in vars(layer).items():
        if value is None or not name.startswith("d") or not name.endswith("_"):
            continue
        if name not in GRAD_NAMES and name[1:] not in param_names:
            continue
        if not isinstance(value, (np.ndarray, np.generic, float, int)):
            continue
        yield name, np.asarray(value, dtype=float)


def clip_gradients(layers, max_norm):
    max_norm = float(max_norm)
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive")

    grads = [value for layer in layers for _, value in _grad_items(layer)]

    if not grads:
        return 0.0, 1.0

    total_norm = float(np.sqrt(sum(np.sum(np.asarray(grad, dtype=float) ** 2) for grad in grads)))
    if total_norm <= max_norm:
        return total_norm, 1.0

    scale = max_norm / (total_norm + 1e-12)
    for layer in layers:
        for name, value in _grad_items(layer):
            setattr(layer, name, value * scale)
    return total_norm, scale


def accumulate_gradients(layers, store):
    """Add current layer gradients to an accumulation mapping."""
    for layer_idx, layer in enumerate(layers):
        for name, value in _grad_items(layer):
            key = (layer_idx, name)
            if key in store:
                store[key] += value
            else:
                store[key] = value.copy()


def load_accumulated_gradients(layers, store, count):
    """Replace current gradients with the averaged accumulated gradients."""
    count = int(count)
    if count < 1:
        raise ValueError("count must be positive")
    for (layer_idx, name), value in store.items():
        setattr(layers[layer_idx], name, value / count)


clip_grads = clip_gradients
