"""Composable streaming signal monitoring workflows."""

from __future__ import annotations

from typing import Any

import numpy as np


class SignalMonitor:
    """Run streaming feature extraction and event detection as one workflow.

    ``push`` and ``flush`` return dictionaries with ``features`` and ``events``
    arrays. Integer event positions are converted to positions in the complete
    feature stream, so callers can safely concatenate results from chunks.
    """

    def __init__(self, feature_extractor, detector=None):
        if not hasattr(feature_extractor, "push") or not hasattr(feature_extractor, "flush"):
            raise TypeError("feature_extractor must define push() and flush()")
        if detector is not None and not hasattr(detector, "transform"):
            raise TypeError("detector must define transform()")
        self.feature_extractor = feature_extractor
        self.detector = detector
        self.frames_seen_ = 0

    def _detect(self, features):
        features = np.asarray(features, dtype=float)
        if features.size == 0 or self.detector is None:
            return np.empty(0, dtype=int)
        events = np.asarray(self.detector.transform(features))
        if np.issubdtype(events.dtype, np.integer):
            events = events.astype(int, copy=False) + self.frames_seen_
        return events

    def _result(self, features):
        features = np.asarray(features, dtype=float)
        events = self._detect(features)
        if features.ndim == 1 and features.size:
            n_frames = 1
        elif features.ndim >= 2:
            n_frames = features.shape[0]
        else:
            n_frames = 0
        self.frames_seen_ += n_frames
        return {"features": features, "events": events}

    def push(self, chunk):
        """Process one raw signal chunk and return newly produced results."""

        return self._result(self.feature_extractor.push(chunk))

    def flush(self):
        """Flush the feature extractor and return the final results."""

        return self._result(self.feature_extractor.flush())

    def reset(self):
        """Reset both the underlying stream and global feature positions."""

        self.feature_extractor.reset()
        self.frames_seen_ = 0
        return self

    def state(self) -> dict[str, Any]:
        """Return lightweight monitoring state for logging or dashboards."""

        return {"frames_seen": int(self.frames_seen_), "has_detector": self.detector is not None}


__all__ = ["SignalMonitor"]
