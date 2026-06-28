from __future__ import annotations

import math
import numpy as np

from ..base import BaseEstimator
from ..utils import as_2d, check_2d_array
from .isolation_tree import IsolationTree, average_path_length


class IsolationForest(BaseEstimator):
    """Isolation forest for unsupervised anomaly detection."""

    def __init__(
        self,
        n_estimators=100,
        max_samples="auto",
        max_features=1.0,
        contamination=0.1,
        random_state=None,
    ):
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.max_features = max_features
        self.contamination = contamination
        self.random_state = random_state
        self.trees_ = []
        self.cols_ = []
        self.max_samples_ = None
        self.offset_ = None

    def _resolve_max_samples(self, n_samples):
        if self.max_samples == "auto":
            return min(256, n_samples)
        if isinstance(self.max_samples, float):
            if not 0.0 < self.max_samples <= 1.0:
                raise ValueError("max_samples as float must be in (0, 1]")
            return max(1, int(round(self.max_samples * n_samples)))
        value = int(self.max_samples)
        if value < 1 or value > n_samples:
            raise ValueError("max_samples must be between 1 and n_samples")
        return value

    def _resolve_max_features(self, n_features):
        if isinstance(self.max_features, float):
            if not 0.0 < self.max_features <= 1.0:
                raise ValueError("max_features as float must be in (0, 1]")
            return max(1, int(round(self.max_features * n_features)))
        value = int(self.max_features)
        if value < 1 or value > n_features:
            raise ValueError("max_features must be between 1 and n_features")
        return value

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        n_samples, n_features = X.shape
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if not 0.0 < float(self.contamination) < 0.5:
            raise ValueError("contamination must be in (0, 0.5)")

        self.max_samples_ = self._resolve_max_samples(n_samples)
        n_cols = self._resolve_max_features(n_features)
        depth = max(1, int(math.ceil(math.log2(max(self.max_samples_, 2)))))

        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        self.cols_ = []

        for _ in range(self.n_estimators):
            rows = rng.choice(n_samples, size=self.max_samples_, replace=False)
            cols = np.sort(rng.choice(n_features, size=n_cols, replace=False))
            tree = IsolationTree(max_depth=depth, random_state=int(rng.integers(0, 1_000_000_000)))
            tree.fit(X[rows][:, cols])
            self.trees_.append(tree)
            self.cols_.append(cols)

        scores = self.score_samples(X)
        self.offset_ = float(np.quantile(scores, 1.0 - float(self.contamination)))
        return self

    def score_samples(self, X):
        if not self.trees_:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X).astype(float)
        lengths = []
        for tree, cols in zip(self.trees_, self.cols_):
            lengths.append(tree.path_lengths(X[:, cols]))
        mean_length = np.mean(np.vstack(lengths), axis=0)
        scale = average_path_length(self.max_samples_)
        scores = 2.0 ** (-mean_length / max(scale, 1e-12))
        return float(scores[0]) if scores.shape[0] == 1 else scores

    def decision_function(self, X):
        scores = self.score_samples(X)
        out = self.offset_ - scores
        return float(out) if np.ndim(out) == 0 else out

    def predict(self, X):
        scores = self.score_samples(X)
        pred = np.where(scores >= self.offset_, -1, 1)
        return int(pred) if np.ndim(pred) == 0 else pred
