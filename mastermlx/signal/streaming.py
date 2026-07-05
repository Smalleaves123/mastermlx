from __future__ import annotations

import numpy as np

from .core import frame_signal


class SignalChunkBuffer:
    """Accumulate streaming signal chunks until a downstream consumer pulls them."""

    def __init__(self):
        self._buffer = np.empty(0, dtype=float)

    def __len__(self):
        return int(self._buffer.size)

    def append(self, chunk):
        chunk = np.asarray(chunk, dtype=float).ravel()
        if chunk.size == 0:
            return self
        self._buffer = np.concatenate([self._buffer, chunk])
        return self

    def extend(self, chunks):
        for chunk in chunks:
            self.append(chunk)
        return self

    def pop(self, size=None):
        if self._buffer.size == 0:
            return np.empty(0, dtype=float)
        if size is None or int(size) >= self._buffer.size:
            out = self._buffer.copy()
            self.clear()
            return out
        size = int(size)
        if size < 0:
            raise ValueError("size must be non-negative")
        out = self._buffer[:size].copy()
        self._buffer = self._buffer[size:]
        return out

    def peek(self):
        return self._buffer.copy()

    def clear(self):
        self._buffer = np.empty(0, dtype=float)
        return self

    def to_array(self):
        return self._buffer.copy()


class SlidingWindowStream:
    """Stateful sliding-window frame generator for chunked signal input."""

    def __init__(self, frame_length, hop_length=None):
        self.frame_length = int(frame_length)
        self.hop_length = self.frame_length if hop_length is None else int(hop_length)
        if self.frame_length < 1 or self.hop_length < 1:
            raise ValueError("frame_length and hop_length must be at least 1")
        self._buffer = np.empty(0, dtype=float)

    def push(self, chunk):
        chunk = np.asarray(chunk, dtype=float).ravel()
        if chunk.size == 0:
            return np.empty((0, self.frame_length), dtype=float)

        x = np.concatenate([self._buffer, chunk])
        if x.size < self.frame_length:
            self._buffer = x
            return np.empty((0, self.frame_length), dtype=float)

        n_frames = 1 + (x.size - self.frame_length) // self.hop_length
        frames = np.empty((n_frames, self.frame_length), dtype=float)
        for i in range(n_frames):
            start = i * self.hop_length
            frames[i] = x[start : start + self.frame_length]

        consumed = n_frames * self.hop_length
        self._buffer = x[consumed:]
        return frames

    def flush(self, pad_end=False):
        if self._buffer.size == 0:
            return np.empty((0, self.frame_length), dtype=float)
        if not pad_end and self._buffer.size < self.frame_length:
            out = np.empty((0, self.frame_length), dtype=float)
            self._buffer = np.empty(0, dtype=float)
            return out
        if pad_end and self._buffer.size < self.frame_length:
            padded = np.pad(self._buffer, (0, self.frame_length - self._buffer.size))
            self._buffer = np.empty(0, dtype=float)
            return padded.reshape(1, -1)
        return self.push(np.empty(0, dtype=float))


class StreamingFeatureExtractor:
    """Extract fixed features from chunked 1D signals.

    The extractor accumulates samples until a full frame is available and then
    applies a user-supplied feature function to each frame.
    """

    def __init__(self, feature_fn, frame_length, hop_length=None, pad_end=False):
        if feature_fn is None or not callable(feature_fn):
            raise ValueError("feature_fn must be callable")
        self.feature_fn = feature_fn
        self.frame_length = int(frame_length)
        self.hop_length = self.frame_length if hop_length is None else int(hop_length)
        self.pad_end = bool(pad_end)
        self._stream = SlidingWindowStream(self.frame_length, self.hop_length)

    def reset(self):
        self._stream = SlidingWindowStream(self.frame_length, self.hop_length)
        return self

    def push(self, chunk):
        frames = self._stream.push(chunk)
        if frames.size == 0:
            return np.empty((0,), dtype=float)
        features = [np.asarray(self.feature_fn(frame), dtype=float).ravel() for frame in frames]
        return np.vstack(features)

    def flush(self):
        frames = self._stream.flush(pad_end=self.pad_end)
        if frames.size == 0:
            return np.empty((0,), dtype=float)
        features = [np.asarray(self.feature_fn(frame), dtype=float).ravel() for frame in frames]
        self.reset()
        return np.vstack(features)

    def transform(self, X):
        X = np.asarray(X, dtype=float).ravel()
        if X.size == 0:
            return np.empty((0,), dtype=float)
        frames = frame_signal(X, self.frame_length, self.hop_length, pad_end=self.pad_end)
        if frames.size == 0:
            return np.empty((0,), dtype=float)
        features = [np.asarray(self.feature_fn(frame), dtype=float).ravel() for frame in frames]
        return np.vstack(features)

__all__ = ["SignalChunkBuffer", "SlidingWindowStream", "StreamingFeatureExtractor"]
