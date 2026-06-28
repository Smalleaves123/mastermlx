from __future__ import annotations

import numpy as np


def train_test_split(X, y, test_size=0.25, shuffle=True, random_state=None):
    """Split arrays into train and test subsets.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    test_size : float or int
        If float, represents the proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.
    shuffle : bool
        Whether to shuffle before splitting.
    random_state : int | None
        Seed for reproducibility.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    n_samples = X.shape[0]
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("test_size as a float must be between 0 and 1")
        n_test = int(np.ceil(n_samples * test_size))
    else:
        n_test = int(test_size)

    if not 0 < n_test < n_samples:
        raise ValueError("test_size must leave at least one training and one test sample")

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

