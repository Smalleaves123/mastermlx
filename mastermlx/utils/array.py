from __future__ import annotations

import numpy as np

from .random import resolve_rng


def one_hot(y, n_classes=None, dtype=float):
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if y.size == 0:
        raise ValueError("y must be non-empty")

    if n_classes is None:
        classes = np.unique(y)
    else:
        n_classes = int(n_classes)
        if n_classes < 1:
            raise ValueError("n_classes must be at least 1")
        classes = np.arange(n_classes)

    index = {label: i for i, label in enumerate(classes)}
    out = np.zeros((y.shape[0], classes.shape[0]), dtype=dtype)
    for i, label in enumerate(y):
        if label not in index:
            raise ValueError("y contains labels outside the provided classes")
        out[i, index[label]] = 1
    return out


def shuffle_arrays(*arrays, random_state=None):
    if not arrays:
        raise ValueError("at least one array is required")
    arrays = [np.asarray(arr) for arr in arrays]
    n_samples = arrays[0].shape[0]
    for arr in arrays[1:]:
        if arr.shape[0] != n_samples:
            raise ValueError("all arrays must contain the same number of samples")
    rng = resolve_rng(random_state)
    idx = rng.permutation(n_samples)
    return tuple(arr[idx] for arr in arrays)


def batch_iterator(*arrays, batch_size, shuffle=True, drop_last=False, random_state=None):
    if not arrays:
        raise ValueError("at least one array is required")
    arrays = [np.asarray(arr) for arr in arrays]
    n_samples = arrays[0].shape[0]
    for arr in arrays[1:]:
        if arr.shape[0] != n_samples:
            raise ValueError("all arrays must contain the same number of samples")

    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    indices = np.arange(n_samples)
    if shuffle:
        rng = resolve_rng(random_state)
        rng.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        stop = start + batch_size
        if stop > n_samples and drop_last:
            break
        batch_idx = indices[start:stop]
        yield tuple(arr[batch_idx] for arr in arrays)
