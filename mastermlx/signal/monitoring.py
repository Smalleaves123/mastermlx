"""Composable streaming signal monitoring workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..base import BaseResult
from .streaming import MultiChannelStreamAligner


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
        if hasattr(self.detector, "update"):
            events = np.asarray(self.detector.update(features))
            return events
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
        return BaseResult({"features": features, "events": events})

    def push(self, chunk):
        """Process one raw signal chunk and return newly produced results."""

        return self._result(self.feature_extractor.push(chunk))

    def flush(self):
        """Flush the feature extractor and return the final results."""

        return self._result(self.feature_extractor.flush())

    def reset(self):
        """Reset both the underlying stream and global feature positions."""

        self.feature_extractor.reset()
        if self.detector is not None and hasattr(self.detector, "reset"):
            self.detector.reset()
        self.frames_seen_ = 0
        return self

    def state(self) -> dict[str, Any]:
        """Return lightweight monitoring state for logging or dashboards."""

        return BaseResult({"frames_seen": int(self.frames_seen_), "has_detector": self.detector is not None})


class MultiChannelSignalMonitor:
    """Monitor synchronized sensor streams with quality-aware health events.

    The monitor aligns each channel to a target sample rate, emits shared
    sliding windows, calculates channel quality and feature drift, and fuses
    channel scores into one health score. A stable baseline is learned from
    ``baseline_frames`` valid windows. State transitions are debounced through
    ``confirmation_frames`` and ``recovery_frames`` to prevent alert flapping.
    """

    def __init__(
        self,
        channel_names,
        sample_rate,
        feature_fn=None,
        frame_length=256,
        hop_length=None,
        source_sample_rates=None,
        max_interpolation_gap=None,
        min_finite_ratio=1.0,
        saturation_levels=None,
        max_saturation_ratio=0.0,
        min_std=None,
        fusion_weights=None,
        baseline_frames=20,
        baseline_std_floor=1e-6,
        adaptation_rate=0.01,
        drift_threshold=3.0,
        warning_threshold=70.0,
        critical_threshold=40.0,
        confirmation_frames=2,
        recovery_frames=3,
        pad_end=False,
    ):
        self.aligner = MultiChannelStreamAligner(
            channel_names,
            target_sample_rate=sample_rate,
            source_sample_rates=source_sample_rates,
            max_interpolation_gap=max_interpolation_gap,
        )
        self.channel_names = self.aligner.channel_names
        self.sample_rate = self.aligner.target_sample_rate
        self.feature_fn = self._default_features if feature_fn is None else feature_fn
        if not callable(self.feature_fn):
            raise TypeError("feature_fn must be callable")
        self.frame_length = int(frame_length)
        self.hop_length = self.frame_length if hop_length is None else int(hop_length)
        if self.frame_length < 1 or self.hop_length < 1:
            raise ValueError("frame_length and hop_length must be at least 1")
        self.min_finite_ratio = self._validate_fraction(min_finite_ratio, "min_finite_ratio")
        self.max_saturation_ratio = self._validate_fraction(max_saturation_ratio, "max_saturation_ratio")
        self.saturation_levels = self._resolve_saturation_levels(saturation_levels)
        self.min_std = self._validate_positive_or_none(min_std, "min_std")
        self.fusion_weights = self._resolve_fusion_weights(fusion_weights)
        self.baseline_frames = self._validate_count(baseline_frames, "baseline_frames")
        self.baseline_std_floor = self._validate_positive_or_none(baseline_std_floor, "baseline_std_floor")
        self.adaptation_rate = self._validate_fraction(adaptation_rate, "adaptation_rate")
        self.drift_threshold = self._validate_positive_or_none(drift_threshold, "drift_threshold")
        self.warning_threshold = self._validate_score(warning_threshold, "warning_threshold")
        self.critical_threshold = self._validate_score(critical_threshold, "critical_threshold")
        if self.critical_threshold >= self.warning_threshold:
            raise ValueError("critical_threshold must be lower than warning_threshold")
        self.confirmation_frames = self._validate_count(confirmation_frames, "confirmation_frames")
        self.recovery_frames = self._validate_count(recovery_frames, "recovery_frames")
        self.pad_end = bool(pad_end)
        self.reset()

    @staticmethod
    def _validate_fraction(value, name):
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and within [0, 1]")
        return value

    @staticmethod
    def _validate_positive_or_none(value, name):
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite or None")
        return value

    @staticmethod
    def _validate_count(value, name):
        value = int(value)
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
        return value

    @staticmethod
    def _validate_score(value, name):
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value <= 100.0:
            raise ValueError(f"{name} must be finite and within [0, 100]")
        return value

    def _resolve_saturation_levels(self, values):
        if values is None:
            return {channel: np.inf for channel in self.channel_names}
        if isinstance(values, Mapping):
            unknown = set(values) - set(self.channel_names)
            if unknown:
                raise KeyError(f"saturation_levels contains unknown channels: {sorted(unknown)!r}")
            return {
                channel: self._validate_positive_or_none(values.get(channel), f"saturation_levels[{channel!r}]")
                or np.inf
                for channel in self.channel_names
            }
        level = self._validate_positive_or_none(values, "saturation_levels")
        return {channel: level for channel in self.channel_names}

    def _resolve_fusion_weights(self, values):
        if values is None:
            weights = {channel: 1.0 for channel in self.channel_names}
        elif isinstance(values, Mapping):
            unknown = set(values) - set(self.channel_names)
            if unknown:
                raise KeyError(f"fusion_weights contains unknown channels: {sorted(unknown)!r}")
            weights = {channel: float(values.get(channel, 1.0)) for channel in self.channel_names}
        else:
            raise TypeError("fusion_weights must be a mapping of channel names to non-negative weights")
        if any(not np.isfinite(weight) or weight < 0.0 for weight in weights.values()):
            raise ValueError("fusion_weights must be finite and non-negative")
        if sum(weights.values()) <= 0.0:
            raise ValueError("fusion_weights must contain at least one positive weight")
        return weights

    @staticmethod
    def _default_features(frame):
        return {
            "mean": float(np.mean(frame)),
            "std": float(np.std(frame)),
            "rms": float(np.sqrt(np.mean(frame * frame))),
            "peak": float(np.max(np.abs(frame))),
        }

    def reset(self):
        """Reset buffered data, learned baseline, and alert lifecycle state."""

        self.aligner.reset()
        self._sample_buffer = np.empty((0, len(self.channel_names)), dtype=float)
        self._time_buffer = np.empty(0, dtype=float)
        self._baseline_rows = []
        self.baseline_mean_ = None
        self.baseline_std_ = None
        self.feature_names_ = None
        self.frames_seen_ = 0
        self.status_ = "initializing"
        self._pending_status = None
        self._pending_count = 0
        return self

    def _quality(self, frame, channel):
        finite = np.isfinite(frame)
        finite_values = frame[finite]
        finite_ratio = float(np.mean(finite))
        level = self.saturation_levels[channel]
        saturation_ratio = float(np.mean(np.abs(frame[finite]) >= level)) if finite_values.size else 0.0
        frozen = bool(
            self.min_std is not None
            and finite_values.size > 1
            and float(np.std(finite_values)) < self.min_std
        )
        reasons = []
        if finite_ratio < self.min_finite_ratio:
            reasons.append("missing_samples")
        if saturation_ratio > self.max_saturation_ratio:
            reasons.append("saturation")
        if frozen:
            reasons.append("frozen_signal")
        valid = not reasons
        score = 100.0 * finite_ratio * (1.0 - saturation_ratio)
        if frozen:
            score = 0.0
        return BaseResult(
            {
                "channel": channel,
                "valid": valid,
                "status": "valid" if valid else "invalid",
                "finite_ratio": finite_ratio,
                "missing_samples": int(frame.size - finite_values.size),
                "saturation_ratio": saturation_ratio,
                "frozen": frozen,
                "score": float(score),
                "reasons": tuple(reasons),
            }
        )

    def _feature_values(self, frame):
        result = self.feature_fn(frame)
        if isinstance(result, Mapping):
            names = tuple(str(name) for name in result)
            values = np.asarray([result[name] for name in result], dtype=float).ravel()
        else:
            values = np.asarray(result, dtype=float).ravel()
            names = tuple(f"feature_{index}" for index in range(values.size))
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("feature_fn must return at least one finite numeric feature")
        if self.feature_names_ is None:
            self.feature_names_ = names
        elif names != self.feature_names_:
            raise ValueError("feature_fn must return a stable feature layout for every channel and window")
        return values

    def _frame_features(self, frame, quality):
        rows = []
        for index in range(len(self.channel_names)):
            values = frame[:, index]
            if quality[index]["valid"] and np.all(np.isfinite(values)):
                rows.append(self._feature_values(values))
            else:
                rows.append(None)
        if self.feature_names_ is None:
            return np.empty((len(self.channel_names), 0), dtype=float)
        return np.vstack(
            [
                np.full(len(self.feature_names_), np.nan, dtype=float) if row is None else row
                for row in rows
            ]
        )

    def _raw_status(self, health_score):
        if health_score < self.critical_threshold:
            return "critical"
        if health_score < self.warning_threshold:
            return "warning"
        return "healthy"

    def _update_lifecycle(self, raw_status, health_score, timestamp):
        if raw_status == self.status_:
            self._pending_status = None
            self._pending_count = 0
            return None
        if raw_status != self._pending_status:
            self._pending_status = raw_status
            self._pending_count = 1
        else:
            self._pending_count += 1
        required = self.recovery_frames if raw_status == "healthy" else self.confirmation_frames
        if self._pending_count < required:
            return None
        previous = self.status_
        self.status_ = raw_status
        self._pending_status = None
        self._pending_count = 0
        event_type = "recovered" if raw_status == "healthy" else "alert"
        return BaseResult(
            {
                "type": event_type,
                "timestamp": float(timestamp),
                "from_status": previous,
                "to_status": raw_status,
                "health_score": float(health_score),
            }
        )

    def _update_baseline(self, vector):
        if self.baseline_mean_ is None:
            self._baseline_rows.append(vector.copy())
            if len(self._baseline_rows) < self.baseline_frames:
                return False
            values = np.vstack(self._baseline_rows)
            self.baseline_mean_ = np.mean(values, axis=0)
            self.baseline_std_ = np.maximum(np.std(values, axis=0), self.baseline_std_floor)
            self.status_ = "healthy"
            return True
        return False

    def _adapt_baseline(self, vector):
        if self.adaptation_rate == 0.0:
            return
        previous_mean = self.baseline_mean_
        updated_mean = (1.0 - self.adaptation_rate) * previous_mean + self.adaptation_rate * vector
        variance = self.baseline_std_ * self.baseline_std_
        updated_variance = (1.0 - self.adaptation_rate) * variance + self.adaptation_rate * (vector - updated_mean) ** 2
        self.baseline_mean_ = updated_mean
        self.baseline_std_ = np.maximum(np.sqrt(updated_variance), self.baseline_std_floor)

    def _score_frame(self, features, quality):
        quality_scores = np.asarray([item["score"] for item in quality], dtype=float)
        if self.baseline_mean_ is None or not np.all(np.isfinite(features)):
            return quality_scores, float("nan"), None
        deviations = np.abs((features.ravel() - self.baseline_mean_) / self.baseline_std_).reshape(features.shape)
        drift_scores = 100.0 * np.clip(1.0 - np.mean(deviations, axis=1) / self.drift_threshold, 0.0, 1.0)
        channel_scores = quality_scores * drift_scores / 100.0
        weights = np.asarray([self.fusion_weights[channel] for channel in self.channel_names], dtype=float)
        return channel_scores, float(np.average(channel_scores, weights=weights)), deviations

    def _frames_from_aligned(self, samples, timestamps):
        if samples.size == 0:
            return np.empty((0, self.frame_length, len(self.channel_names))), np.empty(0, dtype=float)
        self._sample_buffer = np.vstack([self._sample_buffer, samples])
        self._time_buffer = np.concatenate([self._time_buffer, timestamps])
        if self._sample_buffer.shape[0] < self.frame_length:
            return np.empty((0, self.frame_length, len(self.channel_names))), np.empty(0, dtype=float)
        n_frames = 1 + (self._sample_buffer.shape[0] - self.frame_length) // self.hop_length
        starts = self.hop_length * np.arange(n_frames, dtype=int)
        frames = np.stack([self._sample_buffer[start : start + self.frame_length] for start in starts])
        frame_times = self._time_buffer[starts + self.frame_length - 1]
        consumed = n_frames * self.hop_length
        self._sample_buffer = self._sample_buffer[consumed:]
        self._time_buffer = self._time_buffer[consumed:]
        return frames, frame_times

    def _result(self, frames, frame_times, alignment):
        feature_rows = []
        quality_rows = []
        health_scores = []
        overall_scores = []
        statuses = []
        events = []
        for frame, timestamp in zip(frames, frame_times):
            quality = tuple(self._quality(frame[:, index], channel) for index, channel in enumerate(self.channel_names))
            features = self._frame_features(frame, quality)
            vector = features.ravel()
            valid_baseline_frame = bool(
                features.size and np.all(np.isfinite(vector)) and all(item["valid"] for item in quality)
            )
            if self.baseline_mean_ is None:
                ready = self._update_baseline(vector) if valid_baseline_frame else False
                channel_scores = np.asarray([item["score"] for item in quality], dtype=float)
                health_score = float("nan")
                if ready:
                    events.append(BaseResult({"type": "baseline_ready", "timestamp": float(timestamp)}))
            else:
                channel_scores, health_score, _ = self._score_frame(features, quality)
                raw_status = self._raw_status(health_score)
                event = self._update_lifecycle(raw_status, health_score, timestamp)
                if event is not None:
                    events.append(event)
                if raw_status == "healthy" and valid_baseline_frame:
                    self._adapt_baseline(vector)
            feature_rows.append(features)
            quality_rows.append(quality)
            health_scores.append(channel_scores)
            overall_scores.append(health_score)
            statuses.append(self.status_)
            self.frames_seen_ += 1

        feature_count = 0 if self.feature_names_ is None else len(self.feature_names_)
        features = (
            np.stack(feature_rows)
            if feature_rows
            else np.empty((0, len(self.channel_names), feature_count), dtype=float)
        )
        return BaseResult(
            {
                "timestamps": np.asarray(frame_times, dtype=float),
                "features": features,
                "feature_names": self.feature_names_ or (),
                "quality": tuple(quality_rows),
                "channel_health_scores": np.asarray(health_scores, dtype=float).reshape(len(health_scores), -1),
                "health_scores": np.asarray(overall_scores, dtype=float),
                "statuses": tuple(statuses),
                "events": tuple(events),
                "alignment": alignment,
            }
        )

    def push(self, chunks, timestamps=None):
        """Process one mapping of sensor chunks and return completed windows."""

        alignment = self.aligner.push(chunks, timestamps=timestamps)
        frames, frame_times = self._frames_from_aligned(alignment["samples"], alignment["timestamps"])
        return self._result(frames, frame_times, alignment)

    def flush(self):
        """Flush a padded final window when ``pad_end=True``; otherwise discard it."""

        if self._sample_buffer.shape[0] == 0:
            return self._result(
                np.empty((0, self.frame_length, len(self.channel_names))),
                np.empty(0, dtype=float),
                self.aligner._empty_result(),
            )
        if not self.pad_end:
            self._sample_buffer = np.empty((0, len(self.channel_names)), dtype=float)
            self._time_buffer = np.empty(0, dtype=float)
            return self._result(
                np.empty((0, self.frame_length, len(self.channel_names))),
                np.empty(0, dtype=float),
                self.aligner._empty_result(),
            )
        missing = self.frame_length - self._sample_buffer.shape[0]
        padded = np.pad(self._sample_buffer, ((0, missing), (0, 0)), constant_values=np.nan)
        last_time = self._time_buffer[-1]
        frame_time = last_time + missing / self.sample_rate
        self._sample_buffer = np.empty((0, len(self.channel_names)), dtype=float)
        self._time_buffer = np.empty(0, dtype=float)
        return self._result(padded[None, ...], np.asarray([frame_time]), self.aligner._empty_result())

    def state(self) -> dict[str, Any]:
        """Return alignment, baseline, and alert lifecycle state."""

        return BaseResult(
            {
                "frames_seen": int(self.frames_seen_),
                "status": self.status_,
                "baseline_ready": self.baseline_mean_ is not None,
                "baseline_frames_seen": int(len(self._baseline_rows)),
                "pending_status": self._pending_status,
                "pending_count": int(self._pending_count),
                "alignment": self.aligner.state(),
            }
        )


__all__ = ["MultiChannelSignalMonitor", "SignalMonitor"]
