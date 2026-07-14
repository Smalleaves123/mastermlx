from __future__ import annotations

import numpy as np


def resolve_rng(random_state=None):
    """Return a NumPy generator from an integer, generator, or legacy state."""

    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        seed = int(random_state.randint(0, np.iinfo(np.uint32).max))
        return np.random.default_rng(seed)
    return np.random.default_rng(random_state)


def set_seed(seed: int | None = None):
    np.random.seed(seed)
    return resolve_rng(seed)


def create_rng(seed: int | None = None):
    return resolve_rng(seed)


def shuffle_indices(n, random_state=None):
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = resolve_rng(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    return idx
