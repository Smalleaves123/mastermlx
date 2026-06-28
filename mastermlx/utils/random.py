from __future__ import annotations

import numpy as np


def set_seed(seed: int | None = None):
    np.random.seed(seed)
    return np.random.default_rng(seed)


def create_rng(seed: int | None = None):
    return np.random.default_rng(seed)


def shuffle_indices(n, random_state=None):
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    return idx
