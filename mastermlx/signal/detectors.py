from __future__ import annotations

import numpy as np

from ..accel.signal_ops import online_cusum
from .transforms import SignalTransformer


class PeakDetector(SignalTransformer):
    """Detect simple local maxima above a threshold."""

    def __init__(self, threshold=0.0, min_distance=1):
        self.threshold = float(threshold)
        self.min_distance = int(min_distance)

    def transform(self, X):
        x = np.asarray(X, dtype=float).ravel()
        if x.size < 3:
            return np.empty(0, dtype=int)
        candidates = np.where((x[1:-1] >= x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] >= self.threshold))[0] + 1
        if candidates.size == 0 or self.min_distance <= 1:
            return candidates
        selected = [int(candidates[0])]
        for idx in candidates[1:]:
            if idx - selected[-1] >= self.min_distance:
                selected.append(int(idx))
        return np.asarray(selected, dtype=int)


class EnergyThresholdDetector(SignalTransformer):
    """Flag frames whose RMS energy crosses a threshold."""

    def __init__(self, threshold):
        self.threshold = float(threshold)

    def transform(self, X):
        x = np.asarray(X, dtype=float)
        if x.ndim == 1:
            energy = np.sqrt(np.mean(x * x))
            return bool(energy >= self.threshold)
        if x.ndim == 2:
            energy = np.sqrt(np.mean(x * x, axis=1))
            return energy >= self.threshold
        raise ValueError("X must be a 1D or 2D array")


class CUSUMDetector(SignalTransformer):
    """Detect sustained mean shifts with a simple CUSUM statistic."""

    def __init__(self, threshold=5.0, drift=0.0, baseline_window=20):
        self.threshold = float(threshold)
        self.drift = float(drift)
        self.baseline_window = int(baseline_window)

    def transform(self, X):
        x = np.asarray(X, dtype=float).ravel()
        if x.size == 0:
            return np.empty(0, dtype=int)
        baseline_window = max(1, min(self.baseline_window, x.size))
        mean = float(np.mean(x[:baseline_window]))
        gp = 0.0
        gn = 0.0
        events = []
        for i, value in enumerate(x[baseline_window:], start=baseline_window):
            s = value - mean - self.drift
            gp = max(0.0, gp + s)
            gn = min(0.0, gn + s)
            if gp >= self.threshold or -gn >= self.threshold:
                events.append(i)
                gp = 0.0
                gn = 0.0
        return np.asarray(events, dtype=int)


class OnlineCUSUMDetector(SignalTransformer):
    """Stateful CUSUM detector for chunked scalar observations.

    The first ``baseline_window`` observations establish the reference mean.
    Subsequent calls accumulate the positive and negative CUSUM statistics and
    return event positions in the complete observation stream.
    """

    def __init__(self, threshold=5.0, drift=0.0, baseline_window=20, cooldown=0):
        self.threshold = float(threshold)
        self.drift = float(drift)
        self.baseline_window = int(baseline_window)
        self.cooldown = int(cooldown)
        if self.threshold <= 0.0:
            raise ValueError("threshold must be positive")
        if self.baseline_window < 1:
            raise ValueError("baseline_window must be at least 1")
        if self.cooldown < 0:
            raise ValueError("cooldown must be non-negative")
        self.reset()

    def reset(self):
        self.samples_seen_ = 0
        self.baseline_ = []
        self.baseline_mean_ = None
        self.positive_ = 0.0
        self.negative_ = 0.0
        self.cooldown_left_ = 0
        return self

    def update(self, X):
        values = np.asarray(X, dtype=float).ravel()
        if values.size == 0:
            return np.empty(0, dtype=int)
        if not np.all(np.isfinite(values)):
            raise ValueError("X must contain only finite values")

        result = online_cusum(
            values,
            np.asarray(self.baseline_, dtype=float),
            self.baseline_mean_,
            self.positive_,
            self.negative_,
            self.samples_seen_,
            self.threshold,
            self.drift,
            self.baseline_window,
            self.cooldown_left_,
            self.cooldown,
        )
        events, baseline, mean, positive, negative, samples_seen, cooldown_left = result
        self.baseline_ = np.asarray(baseline, dtype=float).tolist()
        self.baseline_mean_ = None if mean is None or np.isnan(mean) else float(mean)
        self.positive_ = float(positive)
        self.negative_ = float(negative)
        self.samples_seen_ = int(samples_seen)
        self.cooldown_left_ = int(cooldown_left)
        return np.asarray(events, dtype=int)

    def transform(self, X):
        """Alias for ``update`` to remain compatible with signal transformers."""

        return self.update(X)

    def state(self):
        return {
            "samples_seen": int(self.samples_seen_),
            "baseline_ready": self.baseline_mean_ is not None,
            "baseline_mean": self.baseline_mean_,
            "positive": float(self.positive_),
            "negative": float(self.negative_),
            "cooldown_left": int(self.cooldown_left_),
        }


__all__ = ["CUSUMDetector", "EnergyThresholdDetector", "OnlineCUSUMDetector", "PeakDetector"]
