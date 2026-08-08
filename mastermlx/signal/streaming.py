from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..base import BaseResult
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

class MultiChannelStreamAligner:
    """Align timestamped sensor chunks onto one uniformly sampled time grid.

    The aligner accepts a mapping from channel name to one-dimensional sample
    arrays. Supply a matching ``timestamps`` mapping when streams use
    independent clocks; otherwise samples use the regular per-channel clock
    defined by ``source_sample_rates``. Only the common observed time span is
    emitted, so future samples are never fabricated. Non-finite values and
    excessive sample gaps remain missing for downstream quality checks.
    """

    def __init__(
        self,
        channel_names,
        target_sample_rate,
        source_sample_rates=None,
        max_interpolation_gap=None,
    ):
        names = tuple(str(name) for name in channel_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("channel_names must contain unique channel names")
        self.channel_names = names
        self.target_sample_rate = self._validate_rate(target_sample_rate, "target_sample_rate")
        self.source_sample_rates = self._resolve_rates(source_sample_rates, "source_sample_rates")
        self.max_interpolation_gap = self._resolve_gaps(max_interpolation_gap)
        self.reset()

    @staticmethod
    def _validate_rate(value, name):
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
        return value

    def _resolve_rates(self, values, name):
        if values is None:
            return {channel: self.target_sample_rate for channel in self.channel_names}
        if isinstance(values, Mapping):
            unknown = set(values) - set(self.channel_names)
            if unknown:
                raise KeyError(f"{name} contains unknown channels: {sorted(unknown)!r}")
            missing = set(self.channel_names) - set(values)
            if missing:
                raise KeyError(f"{name} is missing channels: {sorted(missing)!r}")
            return {
                channel: self._validate_rate(values[channel], f"{name}[{channel!r}]")
                for channel in self.channel_names
            }
        rate = self._validate_rate(values, name)
        return {channel: rate for channel in self.channel_names}

    def _resolve_gaps(self, values):
        if values is None:
            return {channel: 1.5 / self.source_sample_rates[channel] for channel in self.channel_names}
        if isinstance(values, Mapping):
            unknown = set(values) - set(self.channel_names)
            if unknown:
                raise KeyError(f"max_interpolation_gap contains unknown channels: {sorted(unknown)!r}")
            return {
                channel: self._validate_gap(values.get(channel), channel)
                for channel in self.channel_names
            }
        gap = self._validate_gap(values, "max_interpolation_gap")
        return {channel: gap for channel in self.channel_names}

    @staticmethod
    def _validate_gap(value, name):
        if value is None:
            return np.inf
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"max_interpolation_gap[{name!r}] must be positive and finite or None")
        return value

    def reset(self):
        """Discard buffered samples and restart implicit clocks at zero."""

        self._times = {channel: np.empty(0, dtype=float) for channel in self.channel_names}
        self._values = {channel: np.empty(0, dtype=float) for channel in self.channel_names}
        self._implicit_next_time = {channel: 0.0 for channel in self.channel_names}
        self._next_time = None
        self.samples_emitted_ = 0
        return self

    def _empty_result(self):
        return BaseResult(
            {
                "timestamps": np.empty(0, dtype=float),
                "samples": np.empty((0, len(self.channel_names)), dtype=float),
                "channel_names": self.channel_names,
            }
        )

    def _append(self, channel, values, times):
        if values.size == 0:
            return
        previous = self._times[channel]
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            raise ValueError(f"timestamps[{channel!r}] must be strictly increasing and finite")
        if previous.size and times[0] <= previous[-1]:
            raise ValueError(f"timestamps[{channel!r}] must advance beyond earlier chunks")
        self._times[channel] = np.concatenate([previous, times])
        self._values[channel] = np.concatenate([self._values[channel], values])

    def _interpolate_channel(self, channel, grid):
        times = self._times[channel]
        values = self._values[channel]
        output = np.full(grid.size, np.nan, dtype=float)
        right = np.searchsorted(times, grid, side="left")
        left = right - 1
        exact = np.zeros(grid.size, dtype=bool)
        right_positions = np.flatnonzero(right < times.size)
        if right_positions.size:
            exact[right_positions] = np.isclose(
                times[right[right_positions]], grid[right_positions], rtol=0.0, atol=1e-12
            )
        left_positions = np.flatnonzero(~exact & (left >= 0))
        if left_positions.size:
            matched = np.isclose(times[left[left_positions]], grid[left_positions], rtol=0.0, atol=1e-12)
            exact[left_positions[matched]] = True
            right[left_positions[matched]] = left[left_positions[matched]]
        if np.any(exact):
            output[exact] = values[right[exact]]

        positions = np.flatnonzero(~exact)
        if positions.size == 0:
            return output
        lower = right[positions] - 1
        upper = right[positions]
        valid = (lower >= 0) & (upper < times.size)
        if not np.any(valid):
            return output
        positions = positions[valid]
        lower = lower[valid]
        upper = upper[valid]
        gap = times[upper] - times[lower]
        valid = (gap <= self.max_interpolation_gap[channel]) & np.isfinite(values[lower]) & np.isfinite(values[upper])
        if np.any(valid):
            positions = positions[valid]
            lower = lower[valid]
            upper = upper[valid]
            fraction = (grid[positions] - times[lower]) / (times[upper] - times[lower])
            output[positions] = values[lower] + fraction * (values[upper] - values[lower])
        return output

    def _trim_buffers(self):
        if self._next_time is None:
            return
        for channel in self.channel_names:
            times = self._times[channel]
            keep_from = max(0, int(np.searchsorted(times, self._next_time, side="left")) - 1)
            self._times[channel] = times[keep_from:]
            self._values[channel] = self._values[channel][keep_from:]

    def _emit(self):
        if any(self._times[channel].size == 0 for channel in self.channel_names):
            return self._empty_result()
        interval = 1.0 / self.target_sample_rate
        if self._next_time is None:
            start = max(self._times[channel][0] for channel in self.channel_names)
            self._next_time = np.ceil(start * self.target_sample_rate - 1e-10) / self.target_sample_rate
        end = min(self._times[channel][-1] for channel in self.channel_names)
        count = int(np.floor((end - self._next_time) * self.target_sample_rate + 1e-10)) + 1
        if count <= 0:
            return self._empty_result()
        timestamps = self._next_time + interval * np.arange(count, dtype=float)
        samples = np.column_stack([self._interpolate_channel(channel, timestamps) for channel in self.channel_names])
        self._next_time = float(timestamps[-1] + interval)
        self.samples_emitted_ += int(timestamps.size)
        self._trim_buffers()
        return BaseResult({"timestamps": timestamps, "samples": samples, "channel_names": self.channel_names})

    def push(self, chunks, timestamps=None):
        """Append sensor chunks and return rows newly aligned across channels."""

        if not isinstance(chunks, Mapping):
            raise TypeError("chunks must be a mapping of channel names to 1D arrays")
        unknown = set(chunks) - set(self.channel_names)
        if unknown:
            raise KeyError(f"chunks contains unknown channels: {sorted(unknown)!r}")
        if timestamps is not None and not isinstance(timestamps, Mapping):
            raise TypeError("timestamps must be a mapping when provided")
        if timestamps is not None:
            unknown = set(timestamps) - set(self.channel_names)
            if unknown:
                raise KeyError(f"timestamps contains unknown channels: {sorted(unknown)!r}")

        for channel, chunk in chunks.items():
            values = np.asarray(chunk, dtype=float)
            if values.ndim != 1:
                raise ValueError(f"chunks[{channel!r}] must be a 1D array")
            if timestamps is not None and channel in timestamps:
                times = np.asarray(timestamps[channel], dtype=float)
                if times.ndim != 1 or times.shape != values.shape:
                    raise ValueError(f"timestamps[{channel!r}] must match its chunk shape")
            elif timestamps is not None and values.size:
                raise KeyError(f"timestamps is missing non-empty channel {channel!r}")
            else:
                start = self._implicit_next_time[channel]
                times = start + np.arange(values.size, dtype=float) / self.source_sample_rates[channel]
                self._implicit_next_time[channel] = float(start + values.size / self.source_sample_rates[channel])
            self._append(channel, values, times)
        return self._emit()

    def state(self):
        """Return buffered-channel and emitted-sample counters."""

        waiting = tuple(channel for channel in self.channel_names if self._times[channel].size == 0)
        return BaseResult(
            {
                "channel_names": self.channel_names,
                "samples_emitted": int(self.samples_emitted_),
                "waiting_channels": waiting,
                "buffered_samples": {channel: int(self._times[channel].size) for channel in self.channel_names},
            }
        )


__all__ = [
    "MultiChannelStreamAligner",
    "SignalChunkBuffer",
    "SlidingWindowStream",
    "StreamingFeatureExtractor",
]
