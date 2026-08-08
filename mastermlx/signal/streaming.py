from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..base import BaseResult
from .core import frame_signal, hamming_window, hann_window


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


class StreamingIIRFilter:
    """Causal stateful IIR or SOS filter for discontinuous signal streams.

    Pass ``b`` and ``a`` for one transfer function, or pass an ``(n, 6)``
    ``sos`` array ordered as ``[b0, b1, b2, a0, a1, a2]``. A positive
    ``gap_samples`` value or a changed ``sample_rate`` clears dynamic state;
    callers therefore never filter across an unobserved interval.
    """

    def __init__(self, b=None, a=(1.0,), *, sos=None, sample_rate=None):
        if (b is None) == (sos is None):
            raise ValueError("provide exactly one of b or sos")
        self.sample_rate = None if sample_rate is None else self._validate_rate(sample_rate)
        self._sos = None
        self._b = None
        self._a = None
        if sos is not None:
            sections = np.asarray(sos)
            if sections.ndim != 2 or sections.shape[0] == 0 or sections.shape[1] != 6:
                raise ValueError("sos must have shape (n_sections, 6)")
            if not np.all(np.isfinite(sections)) or np.any(sections[:, 3] == 0.0):
                raise ValueError("sos must be finite with non-zero a0 coefficients")
            dtype = np.result_type(sections.dtype, np.float64)
            sections = sections.astype(dtype, copy=False)
            self._sos = sections / sections[:, 3:4]
            self._dtype = self._sos.dtype
        else:
            b = self._coefficients(b, "b")
            a = self._coefficients(a, "a")
            if a[0] == 0.0:
                raise ValueError("a[0] must be non-zero")
            self._dtype = np.result_type(b.dtype, a.dtype, np.float64)
            self._b = b.astype(self._dtype, copy=False) / a[0]
            self._a = a.astype(self._dtype, copy=False) / a[0]
        self.stream_restarts_ = 0
        self.reset()

    @staticmethod
    def _coefficients(values, name):
        values = np.asarray(values)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a non-empty finite 1D array")
        return values

    @staticmethod
    def _validate_rate(value):
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("sample_rate must be positive and finite")
        return value

    @staticmethod
    def _validate_gap(value):
        if int(value) != value or int(value) < 0:
            raise ValueError("gap_samples must be a non-negative integer")
        return int(value)

    def _clear_filter_state(self):
        if self._sos is not None:
            self._zi = np.zeros((self._sos.shape[0], 2), dtype=self._dtype)
            return
        self._x_state = np.zeros(max(0, self._b.size - 1), dtype=self._dtype)
        self._y_state = np.zeros(max(0, self._a.size - 1), dtype=self._dtype)

    def _promote_state(self, dtype):
        dtype = np.result_type(dtype, self._dtype)
        if self._sos is not None:
            self._zi = self._zi.astype(dtype, copy=False)
        else:
            self._x_state = self._x_state.astype(dtype, copy=False)
            self._y_state = self._y_state.astype(dtype, copy=False)

    def reset(self):
        """Start a new continuous stream and clear all counters."""

        self._clear_filter_state()
        self.samples_seen_ = 0
        self.gap_samples_ = 0
        self.gaps_seen_ = 0
        self.last_reset_reason_ = "manual"
        return self

    def handle_gap(self, gap_samples):
        """Record packet loss and clear state before the next observed sample."""

        gap_samples = self._validate_gap(gap_samples)
        if gap_samples:
            self._clear_filter_state()
            self.gap_samples_ += gap_samples
            self.gaps_seen_ += 1
            self.stream_restarts_ += 1
            self.last_reset_reason_ = "gap"
        return self

    def set_sample_rate(self, sample_rate, *, reset=True):
        """Update rate metadata, resetting dynamic state when it changes."""

        sample_rate = self._validate_rate(sample_rate)
        if self.sample_rate is not None and not np.isclose(sample_rate, self.sample_rate):
            if not reset:
                raise ValueError("sample_rate changes require reset=True")
            self._clear_filter_state()
            self.stream_restarts_ += 1
            self.last_reset_reason_ = "sample_rate_change"
        self.sample_rate = sample_rate
        return self

    def _filter_direct(self, values):
        output = np.empty(values.size, dtype=np.result_type(values.dtype, self._dtype))
        self._promote_state(output.dtype)
        for index, value in enumerate(values):
            filtered = self._b[0] * value
            if self._x_state.size:
                filtered += np.dot(self._b[1:], self._x_state)
            if self._y_state.size:
                filtered -= np.dot(self._a[1:], self._y_state)
            if self._x_state.size:
                self._x_state[1:] = self._x_state[:-1]
                self._x_state[0] = value
            if self._y_state.size:
                self._y_state[1:] = self._y_state[:-1]
                self._y_state[0] = filtered
            output[index] = filtered
        return output

    def _filter_sos(self, values):
        output = np.empty(values.size, dtype=np.result_type(values.dtype, self._dtype))
        self._promote_state(output.dtype)
        for index, value in enumerate(values):
            filtered = value
            for section, coefficients in enumerate(self._sos):
                b0, b1, b2, _, a1, a2 = coefficients
                z1, z2 = self._zi[section]
                section_output = b0 * filtered + z1
                self._zi[section, 0] = b1 * filtered - a1 * section_output + z2
                self._zi[section, 1] = b2 * filtered - a2 * section_output
                filtered = section_output
            output[index] = filtered
        return output

    def push(self, chunk, *, sample_rate=None, gap_samples=0):
        """Filter one chunk, clearing state for declared gaps or rate changes."""

        if sample_rate is not None:
            self.set_sample_rate(sample_rate)
        self.handle_gap(gap_samples)
        values = np.asarray(chunk)
        if values.ndim != 1:
            raise ValueError("chunk must be a 1D array")
        if values.size == 0:
            return np.empty(0, dtype=self._dtype)
        if not np.all(np.isfinite(values)):
            raise ValueError("chunk must contain only finite values")
        output = self._filter_sos(values) if self._sos is not None else self._filter_direct(values)
        self.samples_seen_ += int(values.size)
        return np.real_if_close(output)

    def state(self):
        """Return copy-safe filter state and stream discontinuity counters."""

        state = self._zi.copy() if self._sos is not None else BaseResult(
            {"input": self._x_state.copy(), "output": self._y_state.copy()}
        )
        return BaseResult(
            {
                "sample_rate": self.sample_rate,
                "samples_seen": int(self.samples_seen_),
                "gap_samples": int(self.gap_samples_),
                "gaps_seen": int(self.gaps_seen_),
                "stream_restarts": int(self.stream_restarts_),
                "last_reset_reason": self.last_reset_reason_,
                "filter_state": state,
                "uses_sos": self._sos is not None,
            }
        )


class StreamingSTFT:
    """Stateful STFT with chunk-invariant frames and explicit timing metadata.

    Spectra are emitted only after an entire frame arrives. ``flush()`` emits
    the one zero-padded tail frame required by batch :func:`stft` when
    ``pad_end=True``. Declared gaps and sample-rate changes discard buffered
    overlap, so no frame can span a discontinuity.
    """

    def __init__(
        self,
        frame_length=256,
        hop_length=None,
        window="hann",
        n_fft=None,
        sample_rate=None,
        pad_end=True,
    ):
        self.frame_length = int(frame_length)
        self.hop_length = self.frame_length // 2 if hop_length is None else int(hop_length)
        self.n_fft = self.frame_length if n_fft is None else int(n_fft)
        if self.frame_length < 1 or not 1 <= self.hop_length <= self.frame_length:
            raise ValueError("frame_length must be positive and hop_length must be within [1, frame_length]")
        if self.n_fft < self.frame_length:
            raise ValueError("n_fft must be at least frame_length")
        if window == "hann":
            self.window = hann_window(self.frame_length)
        elif window == "hamming":
            self.window = hamming_window(self.frame_length)
        elif window is None:
            self.window = np.ones(self.frame_length, dtype=float)
        else:
            raise ValueError("window must be one of: hann, hamming, None")
        self.window_name = window
        self.sample_rate = None if sample_rate is None else StreamingIIRFilter._validate_rate(sample_rate)
        self.pad_end = bool(pad_end)
        self.stream_restarts_ = 0
        self.reset()

    @staticmethod
    def _validate_gap(value):
        return StreamingIIRFilter._validate_gap(value)

    def _empty_result(self):
        return BaseResult(
            {
                "spectrogram": np.empty((0, self.n_fft // 2 + 1), dtype=complex),
                "frame_start_samples": np.empty(0, dtype=int),
                "frame_end_samples": np.empty(0, dtype=int),
                "frame_end_times": np.empty(0, dtype=float),
            }
        )

    def _clear_frames(self):
        self._buffer = np.empty(0, dtype=float)
        self._buffer_start_sample = self.samples_seen_
        self._last_frame_end = None

    def reset(self):
        """Start a new stream and clear overlap, counters, and gap history."""

        self.samples_seen_ = 0
        self._clear_frames()
        self.gap_samples_ = 0
        self.gaps_seen_ = 0
        self.last_reset_reason_ = "manual"
        return self

    def handle_gap(self, gap_samples):
        """Drop overlap and advance timing by a declared missing interval."""

        gap_samples = self._validate_gap(gap_samples)
        if gap_samples:
            self.samples_seen_ += gap_samples
            self._clear_frames()
            self.gap_samples_ += gap_samples
            self.gaps_seen_ += 1
            self.stream_restarts_ += 1
            self.last_reset_reason_ = "gap"
        return self

    def set_sample_rate(self, sample_rate, *, reset=True):
        """Change sample rate only by discarding the current overlap state."""

        sample_rate = StreamingIIRFilter._validate_rate(sample_rate)
        if self.sample_rate is not None and not np.isclose(sample_rate, self.sample_rate):
            if not reset:
                raise ValueError("sample_rate changes require reset=True")
            self.reset()
            self.stream_restarts_ += 1
            self.last_reset_reason_ = "sample_rate_change"
        self.sample_rate = sample_rate
        return self

    def _format_result(self, frames, starts):
        if frames.size == 0:
            return self._empty_result()
        spectrum = np.fft.rfft(frames * self.window[None, :], n=self.n_fft, axis=1)
        starts = np.asarray(starts, dtype=int)
        ends = starts + self.frame_length - 1
        if self.sample_rate is None:
            times = ends.astype(float)
        else:
            times = ends.astype(float) / self.sample_rate
        return BaseResult(
            {
                "spectrogram": spectrum,
                "frame_start_samples": starts,
                "frame_end_samples": ends,
                "frame_end_times": times,
            }
        )

    def _emit_completed(self):
        if self._buffer.size < self.frame_length:
            return self._empty_result()
        n_frames = 1 + (self._buffer.size - self.frame_length) // self.hop_length
        starts = self._buffer_start_sample + self.hop_length * np.arange(n_frames, dtype=int)
        frames = np.stack(
            [self._buffer[index * self.hop_length : index * self.hop_length + self.frame_length] for index in range(n_frames)]
        )
        self._last_frame_end = int(starts[-1] + self.frame_length - 1)
        consumed = n_frames * self.hop_length
        self._buffer = self._buffer[consumed:]
        self._buffer_start_sample += consumed
        return self._format_result(frames, starts)

    def push(self, chunk, *, sample_rate=None, gap_samples=0):
        """Append samples and return only complete STFT frames."""

        if sample_rate is not None:
            self.set_sample_rate(sample_rate)
        self.handle_gap(gap_samples)
        values = np.asarray(chunk, dtype=float)
        if values.ndim != 1:
            raise ValueError("chunk must be a 1D array")
        if values.size == 0:
            return self._empty_result()
        if not np.all(np.isfinite(values)):
            raise ValueError("chunk must contain only finite values")
        self._buffer = np.concatenate([self._buffer, values])
        self.samples_seen_ += int(values.size)
        return self._emit_completed()

    def flush(self, *, reset=True):
        """Emit the deterministic zero-padded tail frame, then optionally reset."""

        has_tail = self._buffer.size and (
            self._last_frame_end is None or self._last_frame_end < self.samples_seen_ - 1
        )
        if self.pad_end and has_tail:
            start = self._buffer_start_sample
            frame = np.pad(self._buffer, (0, self.frame_length - self._buffer.size))
            result = self._format_result(frame[None, :], np.asarray([start], dtype=int))
            self._last_frame_end = int(start + self.frame_length - 1)
        else:
            result = self._empty_result()
        if reset:
            self.reset()
        return result

    def state(self):
        """Return frame latency and current overlap/discontinuity state."""

        return BaseResult(
            {
                "sample_rate": self.sample_rate,
                "samples_seen": int(self.samples_seen_),
                "buffered_samples": int(self._buffer.size),
                "latency_samples": int(self.frame_length - 1),
                "latency_seconds": None if self.sample_rate is None else (self.frame_length - 1) / self.sample_rate,
                "gap_samples": int(self.gap_samples_),
                "gaps_seen": int(self.gaps_seen_),
                "stream_restarts": int(self.stream_restarts_),
                "last_reset_reason": self.last_reset_reason_,
            }
        )

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
    "StreamingIIRFilter",
    "StreamingSTFT",
]
