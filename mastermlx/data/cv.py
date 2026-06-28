from __future__ import annotations

import numpy as np


class KFold:
    """K-fold cross-validator."""

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = int(n_splits)
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        X = np.asarray(X)
        n_samples = X.shape[0]
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.n_splits > n_samples:
            raise ValueError("n_splits cannot be larger than the number of samples")

        idx = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(idx)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        start = 0
        for fold_size in fold_sizes:
            stop = start + fold_size
            test_idx = idx[start:stop]
            train_idx = np.concatenate([idx[:start], idx[stop:]])
            yield train_idx, test_idx
            start = stop


class StratifiedKFold:
    """Stratified K-fold cross-validator."""

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = int(n_splits)
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        if y.shape[0] != n_samples:
            raise ValueError("X and y must contain the same number of samples")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        idx = np.arange(n_samples)
        rng = np.random.default_rng(self.random_state)
        by_fold = [[] for _ in range(self.n_splits)]

        for cls in np.unique(y):
            cls_idx = idx[y == cls].copy()
            if cls_idx.shape[0] < self.n_splits:
                raise ValueError("Each class must have at least n_splits samples")
            if self.shuffle:
                rng.shuffle(cls_idx)
            parts = np.array_split(cls_idx, self.n_splits)
            for fold, part in enumerate(parts):
                by_fold[fold].extend(part.tolist())

        for fold in range(self.n_splits):
            test_idx = np.array(sorted(by_fold[fold]), dtype=int)
            mask = np.ones(n_samples, dtype=bool)
            mask[test_idx] = False
            train_idx = idx[mask]
            yield train_idx, test_idx


class TimeSeriesSplit:
    """Time series cross-validator with expanding training windows."""

    def __init__(self, n_splits=5, test_size=None):
        self.n_splits = int(n_splits)
        self.test_size = test_size

    def split(self, X, y=None):
        X = np.asarray(X)
        n_samples = X.shape[0]
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        if self.test_size is None:
            fold_size = n_samples // (self.n_splits + 1)
        else:
            fold_size = int(self.test_size)
        if fold_size < 1:
            raise ValueError("test_size must be at least 1")
        if fold_size * (self.n_splits + 1) > n_samples:
            raise ValueError("Not enough samples for the requested number of splits")

        start = n_samples - self.n_splits * fold_size
        for split_idx in range(self.n_splits):
            test_start = start + split_idx * fold_size
            test_stop = test_start + fold_size
            train_idx = np.arange(0, test_start, dtype=int)
            test_idx = np.arange(test_start, test_stop, dtype=int)
            if train_idx.size == 0:
                raise ValueError("The first training fold would be empty")
            yield train_idx, test_idx


class GroupKFold:
    """K-fold cross-validator with non-overlapping groups."""

    def __init__(self, n_splits=5):
        self.n_splits = int(n_splits)

    def split(self, X, y=None, groups=None):
        X = np.asarray(X)
        if groups is None:
            raise ValueError("groups must be provided")
        groups = np.asarray(groups)
        n_samples = X.shape[0]
        if groups.shape[0] != n_samples:
            raise ValueError("X and groups must contain the same number of samples")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        uniq, inv = np.unique(groups, return_inverse=True)
        n_groups = uniq.shape[0]
        if self.n_splits > n_groups:
            raise ValueError("n_splits cannot be larger than the number of groups")

        group_sizes = np.bincount(inv)
        order = np.argsort(group_sizes)[::-1]
        fold_sizes = np.zeros(self.n_splits, dtype=int)
        group_to_fold = np.empty(n_groups, dtype=int)

        for group_idx in order:
            fold = int(np.argmin(fold_sizes))
            group_to_fold[group_idx] = fold
            fold_sizes[fold] += group_sizes[group_idx]

        idx = np.arange(n_samples)
        for fold in range(self.n_splits):
            test_mask = group_to_fold[inv] == fold
            test_idx = idx[test_mask]
            train_idx = idx[~test_mask]
            yield train_idx, test_idx


class ShuffleSplit:
    """Random permutation cross-validator."""

    def __init__(self, n_splits=10, test_size=0.1, train_size=None, random_state=None):
        self.n_splits = int(n_splits)
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def _resolve_sizes(self, n_samples):
        if isinstance(self.test_size, float):
            if not 0.0 < self.test_size < 1.0:
                raise ValueError("test_size as float must be in (0, 1)")
            n_test = int(np.ceil(n_samples * self.test_size))
        else:
            n_test = int(self.test_size)

        if self.train_size is None:
            n_train = n_samples - n_test
        elif isinstance(self.train_size, float):
            if not 0.0 < self.train_size < 1.0:
                raise ValueError("train_size as float must be in (0, 1)")
            n_train = int(np.floor(n_samples * self.train_size))
        else:
            n_train = int(self.train_size)

        if n_train < 1 or n_test < 1 or n_train + n_test > n_samples:
            raise ValueError("train_size and test_size must leave at least one sample in each split")
        return n_train, n_test

    def split(self, X, y=None, groups=None):
        X = np.asarray(X)
        n_samples = X.shape[0]
        if self.n_splits < 1:
            raise ValueError("n_splits must be at least 1")
        n_train, n_test = self._resolve_sizes(n_samples)
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_splits):
            idx = rng.permutation(n_samples)
            test_idx = idx[:n_test]
            train_idx = idx[n_test : n_test + n_train]
            yield train_idx, test_idx


class RepeatedKFold:
    """KFold repeated n_repeats times with different random splits."""

    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        self.n_splits = int(n_splits)
        self.n_repeats = int(n_repeats)
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_repeats):
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=rng.integers(0, 1 << 31))
            yield from kf.split(X, y)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats


class LeaveOneOut:
    """Leave-one-out cross-validator."""

    def split(self, X, y=None, groups=None):
        X = np.asarray(X)
        n = X.shape[0]
        for i in range(n):
            yield np.concatenate([np.arange(i), np.arange(i + 1, n)]), np.array([i])

    def get_n_splits(self, X=None, y=None, groups=None):
        return int(np.asarray(X).shape[0])
