from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils.metrics import r2_score
from ..utils.validation import check_1d_array, check_2d_array, check_same_rows


class RANSACRegressor(BaseEstimator):
    """RANSAC (RANdom SAmple Consensus) robust linear regression."""

    def __init__(self, min_samples=None, residual_threshold=None, max_trials=100,
                 random_state=None):
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = int(max_trials)
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = 0.0
        self.inlier_mask_ = None
        self.n_trials_ = 0

    def fit(self, X, y=None):
        X = check_2d_array(X).astype(float)
        y = check_1d_array(y).astype(float)
        X, y = check_same_rows(X, y)
        n, d = X.shape
        if self.max_trials < 1:
            raise ValueError("max_trials must be at least 1")
        min_s = (
            max(d + 1, int(n * 0.1))
            if self.min_samples is None
            else int(self.min_samples)
        )
        if min_s < 1 or min_s > n:
            raise ValueError("min_samples must be between 1 and the number of samples")
        thresh = (
            np.median(np.abs(y - np.median(y))) * 2.0
            if self.residual_threshold is None
            else float(self.residual_threshold)
        )
        if not np.isfinite(thresh) or thresh < 0.0:
            raise ValueError("residual_threshold must be a finite non-negative value")
        rng = np.random.default_rng(self.random_state)

        best_inliers = 0
        best_coef = None
        best_intercept = 0.0

        Xb = np.column_stack([np.ones(n), X])

        trial = 0
        for trial in range(1, self.max_trials + 1):
            subset = rng.choice(n, size=min_s, replace=False)
            try:
                beta = np.linalg.lstsq(Xb[subset], y[subset], rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            resid = np.abs(y - Xb @ beta)
            inliers = int(np.sum(resid <= thresh))
            if inliers > best_inliers:
                best_inliers = inliers
                best_coef = beta[1:]
                best_intercept = float(beta[0])
                if best_inliers == n:
                    break

        # Refit on all inliers
        if best_coef is not None:
            resid = np.abs(y - Xb @ np.r_[best_intercept, best_coef])
            mask = resid <= thresh
            if np.sum(mask) >= min_s:
                beta = np.linalg.lstsq(Xb[mask], y[mask], rcond=None)[0]
                best_coef = beta[1:]
                best_intercept = float(beta[0])
            self.inlier_mask_ = mask

        if best_coef is None:
            raise RuntimeError("RANSAC failed to find a valid consensus set")
        self.coef_ = best_coef
        self.intercept_ = best_intercept
        self.n_trials_ = trial
        self._set_n_features(X)
        return self

    def predict(self, X):
        X = self._check_X(X, dtype=float)
        if self.coef_ is None:
            raise RuntimeError("Model has not been fit yet")
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        return r2_score(y, self.predict(X))
