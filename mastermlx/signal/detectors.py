from __future__ import annotations

import numpy as np

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


__all__ = ["CUSUMDetector", "EnergyThresholdDetector", "PeakDetector"]
