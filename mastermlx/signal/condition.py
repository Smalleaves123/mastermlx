"""Condition-monitoring helpers for sensor and vibration signals."""

from __future__ import annotations

import numpy as np
from collections.abc import Mapping

from .features import rms_energy, spectral_bandwidth, spectral_centroid, zero_crossing_rate
from .fourier import band_energy, dominant_frequency


def _finite_signal(signal, name="signal"):
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"{name} must contain at least one finite sample")
    return values, finite


def signal_quality_report(signal, sample_rate=None, reference=None, saturation_level=None):
    """Summarize data quality and basic amplitude health indicators.

    Non-finite samples are reported rather than silently discarded from the
    quality counters. Amplitude statistics are computed on finite samples.
    ``reference`` enables optional SNR/PSNR metrics for reconstructed or
    denoised signals.
    """

    values, finite = _finite_signal(signal)
    sample_rate = None if sample_rate is None else float(sample_rate)
    if sample_rate is not None and (not np.isfinite(sample_rate) or sample_rate <= 0.0):
        raise ValueError("sample_rate must be positive and finite")

    mean = float(np.mean(finite))
    std = float(np.std(finite))
    rms = float(np.sqrt(np.mean(finite * finite)))
    peak = float(np.max(np.abs(finite)))
    report = {
        "n_samples": int(values.size),
        "finite_samples": int(finite.size),
        "nan_samples": int(np.count_nonzero(np.isnan(values))),
        "inf_samples": int(np.count_nonzero(np.isinf(values))),
        "finite_ratio": float(finite.size / values.size),
        "valid": bool(finite.size == values.size),
        "mean": mean,
        "std": std,
        "rms": rms,
        "peak": peak,
        "peak_to_peak": float(np.max(finite) - np.min(finite)),
        "crest_factor": float(peak / (rms + 1e-12)),
        "dc_ratio": float(abs(mean) / (rms + 1e-12)),
    }
    if sample_rate is not None:
        report["sample_rate"] = sample_rate
        report["duration"] = float(values.size / sample_rate)
    if saturation_level is not None:
        saturation_level = float(saturation_level)
        if not np.isfinite(saturation_level) or saturation_level <= 0.0:
            raise ValueError("saturation_level must be positive and finite")
        saturated = np.abs(finite) >= saturation_level
        report["saturation_level"] = saturation_level
        report["saturated_samples"] = int(np.count_nonzero(saturated))
        report["saturation_ratio"] = float(np.mean(saturated))
    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        if reference.shape != values.shape:
            raise ValueError("reference must have the same shape as signal")
        if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(values)):
            raise ValueError("signal and reference must be finite for SNR/PSNR")
        noise = values - reference
        noise_power = float(np.mean(noise * noise))
        signal_power = float(np.mean(values * values))
        peak_value = float(np.max(np.abs(values)))
        report["snr_db"] = float(10.0 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12)))
        report["psnr_db"] = float(10.0 * np.log10((peak_value**2 + 1e-12) / (noise_power + 1e-12)))
    return report


def _resolve_analysis_frame_length(n_samples, n_fft=None, frame_length=None):
    n_fft = None if n_fft is None else int(n_fft)
    if n_fft is not None and n_fft < 2:
        raise ValueError("n_fft must be at least 2")
    if frame_length is None:
        frame_length = min(256, n_fft) if n_fft is not None else 256
    frame_length = int(frame_length)
    if frame_length < 2:
        raise ValueError("frame_length must be at least 2")
    if n_fft is not None and n_fft < frame_length:
        raise ValueError("n_fft must be at least frame_length")
    return min(frame_length, max(2, int(n_samples)))


def vibration_features(signal, sample_rate, bands=None, n_fft=None, window="hann", frame_length=None):
    """Extract interpretable time/frequency features for condition monitoring.

    The returned dictionary is intentionally flat so it can be logged as a
    row in a dataframe or passed through a model feature pipeline. Band names
    are generated as ``band_energy_0``, ``band_energy_1``, and so on.
    """

    values, _ = _finite_signal(signal)
    if not np.all(np.isfinite(values)):
        raise ValueError("signal must contain only finite values")
    sample_rate = float(sample_rate)
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive and finite")
    frame_length = _resolve_analysis_frame_length(len(values), n_fft=n_fft, frame_length=frame_length)

    centered = values - np.mean(values)
    std = float(np.std(values))
    rms = rms_energy(values)
    peak = float(np.max(np.abs(values)))
    variance = std**2
    if std > 1e-12:
        skewness = float(np.mean(centered**3) / (std**3))
        kurtosis = float(np.mean(centered**4) / (variance**2))
    else:
        skewness = 0.0
        kurtosis = 0.0

    dominant_hz, dominant_amplitude, _ = dominant_frequency(
        values,
        sample_rate=sample_rate,
        n_fft=n_fft,
        window=window,
    )
    energies = band_energy(
        values,
        sample_rate=sample_rate,
        bands=bands,
        n_fft=n_fft,
        window=window,
        normalize=True,
    )
    centroid = np.asarray(
        spectral_centroid(values, sample_rate=sample_rate, frame_length=frame_length, n_fft=n_fft, window=window),
        dtype=float,
    )
    bandwidth = np.asarray(
        spectral_bandwidth(values, sample_rate=sample_rate, frame_length=frame_length, n_fft=n_fft, window=window),
        dtype=float,
    )
    result = {
        "mean": float(np.mean(values)),
        "std": std,
        "rms": rms,
        "peak": peak,
        "peak_to_peak": float(np.ptp(values)),
        "crest_factor": float(peak / (rms + 1e-12)),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "zero_crossing_rate": zero_crossing_rate(values),
        "dominant_frequency": dominant_hz,
        "dominant_amplitude": dominant_amplitude,
        "spectral_centroid": float(np.mean(centroid)),
        "spectral_bandwidth": float(np.mean(bandwidth)),
    }
    result.update({f"band_energy_{index}": float(value) for index, value in enumerate(energies)})
    return result


def _feature_limits(feature_limits):
    if feature_limits is None:
        return {}
    if not isinstance(feature_limits, Mapping):
        raise TypeError("feature_limits must be a mapping of feature names to (lower, upper)")
    limits = {}
    for name, bounds in feature_limits.items():
        if not isinstance(bounds, (tuple, list, np.ndarray)) or len(bounds) != 2:
            raise ValueError(f"limit for {name!r} must be a (lower, upper) pair")
        lower, upper = bounds
        lower = None if lower is None else float(lower)
        upper = None if upper is None else float(upper)
        if lower is not None and not np.isfinite(lower):
            raise ValueError(f"lower limit for {name!r} must be finite or None")
        if upper is not None and not np.isfinite(upper):
            raise ValueError(f"upper limit for {name!r} must be finite or None")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"lower limit for {name!r} must not exceed upper limit")
        limits[str(name)] = (lower, upper)
    return limits


def assess_signal_health(
    signal,
    sample_rate,
    feature_limits=None,
    bands=None,
    n_fft=None,
    window="hann",
    saturation_level=None,
    frame_length=None,
):
    """Assess one sensor signal with quality metrics and threshold-based alerts.

    ``feature_limits`` maps a feature name to ``(lower, upper)``. Either bound
    may be ``None``. The result is JSON-friendly apart from the nested feature
    values and contains ``status``, ``health_score``, ``alerts``, ``quality``,
    ``features``, and ``violations`` keys.
    """

    limits = _feature_limits(feature_limits)
    quality = signal_quality_report(
        signal,
        sample_rate=sample_rate,
        saturation_level=saturation_level,
    )
    if not quality["valid"]:
        return {
            "status": "critical",
            "health_score": 0.0,
            "alerts": ("non_finite_samples",),
            "quality": quality,
            "features": None,
            "violations": {},
        }

    features = vibration_features(
        signal,
        sample_rate=sample_rate,
        bands=bands,
        n_fft=n_fft,
        window=window,
        frame_length=frame_length,
    )
    violations = {}
    for name, (lower, upper) in limits.items():
        if name not in features:
            raise KeyError(f"feature limit refers to unknown feature {name!r}")
        value = float(features[name])
        if lower is not None and value < lower:
            violations[name] = {"value": value, "bound": "lower", "limit": lower}
        elif upper is not None and value > upper:
            violations[name] = {"value": value, "bound": "upper", "limit": upper}

    alerts = list(violations)
    if quality.get("saturation_ratio", 0.0) > 0.0:
        alerts.append("saturation")
    score = 100.0 if not limits else max(0.0, 100.0 * (1.0 - len(violations) / len(limits)))
    if alerts:
        status = "warning"
    else:
        status = "healthy"
    return {
        "status": status,
        "health_score": float(score),
        "alerts": tuple(alerts),
        "quality": quality,
        "features": features,
        "violations": violations,
    }


def windowed_vibration_features(
    signal,
    sample_rate,
    window_length,
    hop_length=None,
    bands=None,
    n_fft=None,
    window="hann",
    pad_end=False,
    frame_length=None,
):
    """Extract vibration features from overlapping windows of one signal.

    Returns a dictionary with ``start_samples``, ``start_times``,
    ``features`` (a matrix), and ``feature_names``. By default only complete
    windows are used; set ``pad_end=True`` to include a zero-padded final
    window.
    """

    values, _ = _finite_signal(signal)
    if not np.all(np.isfinite(values)):
        raise ValueError("signal must contain only finite values")
    window_length = int(window_length)
    hop_length = window_length if hop_length is None else int(hop_length)
    if window_length < 1:
        raise ValueError("window_length must be at least 1")
    if hop_length < 1:
        raise ValueError("hop_length must be at least 1")

    starts = list(range(0, max(values.size - window_length + 1, 0), hop_length))
    if pad_end and (not starts or starts[-1] + window_length < values.size):
        next_start = 0 if not starts else starts[-1] + hop_length
        if next_start < values.size:
            starts.append(next_start)
    rows = []
    for start in starts:
        chunk = values[start : start + window_length]
        if chunk.size < window_length:
            chunk = np.pad(chunk, (0, window_length - chunk.size))
        rows.append(
            vibration_features(
                chunk,
                sample_rate=sample_rate,
                bands=bands,
                n_fft=n_fft,
                window=window,
                frame_length=frame_length,
            )
        )
    if rows:
        names = tuple(rows[0])
        matrix = np.asarray([[row[name] for name in names] for row in rows], dtype=float)
    else:
        names = tuple(vibration_features(np.zeros(window_length), sample_rate, bands, n_fft, window))
        matrix = np.empty((0, len(names)), dtype=float)
    return {
        "start_samples": np.asarray(starts, dtype=int),
        "start_times": np.asarray(starts, dtype=float) / float(sample_rate),
        "features": matrix,
        "feature_names": names,
    }


class SignalHealthMonitor:
    """Reusable threshold-based health assessor for sensor channels."""

    def __init__(
        self,
        sample_rate,
        feature_limits=None,
        bands=None,
        n_fft=None,
        window="hann",
        saturation_level=None,
        frame_length=None,
    ):
        self.sample_rate = float(sample_rate)
        self.feature_limits = None if feature_limits is None else dict(feature_limits)
        self.bands = None if bands is None else list(bands)
        self.n_fft = None if n_fft is None else int(n_fft)
        self.window = window
        self.saturation_level = saturation_level
        self.frame_length = None if frame_length is None else int(frame_length)

    def assess(self, signal):
        """Return one health assessment dictionary."""

        return assess_signal_health(
            signal,
            sample_rate=self.sample_rate,
            feature_limits=self.feature_limits,
            bands=self.bands,
            n_fft=self.n_fft,
            window=self.window,
            saturation_level=self.saturation_level,
            frame_length=self.frame_length,
        )

    def assess_batch(self, X):
        """Assess a 1D signal or each row of a 2D signal batch."""

        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("X must have shape (n_signals, n_samples) or be 1D")
        return [self.assess(signal) for signal in values]

    def get_params(self, deep=True):
        return {
            "sample_rate": self.sample_rate,
            "feature_limits": self.feature_limits,
            "bands": self.bands,
            "n_fft": self.n_fft,
            "window": self.window,
            "saturation_level": self.saturation_level,
            "frame_length": self.frame_length,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class VibrationFeatureTransformer:
    """Fit/transform wrapper for vibration features.

    A single signal returns a one-dimensional feature vector. A batch of
    equal-length signals returns a two-dimensional matrix. ``feature_names_``
    maps columns back to the dictionary keys from :func:`vibration_features`.
    """

    def __init__(self, sample_rate, bands=None, n_fft=None, window="hann"):
        self.sample_rate = float(sample_rate)
        self.bands = None if bands is None else list(bands)
        self.n_fft = None if n_fft is None else int(n_fft)
        self.window = window
        self.feature_names_ = None

    def fit(self, X=None, y=None):
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("X must have shape (n_signals, n_samples) or be 1D")
        first = vibration_features(
            values[0],
            sample_rate=self.sample_rate,
            bands=self.bands,
            n_fft=self.n_fft,
            window=self.window,
        )
        self.feature_names_ = tuple(first.keys())
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        single = values.ndim == 1
        if single:
            values = values[None, :]
        if values.ndim != 2:
            raise ValueError("X must have shape (n_signals, n_samples) or be 1D")
        if self.feature_names_ is None:
            self.fit(values)
        feature_names = self.feature_names_
        if feature_names is None:  # pragma: no cover - fit() establishes this
            raise RuntimeError("VibrationFeatureTransformer has not been fit")
        rows = []
        for signal in values:
            features = vibration_features(
                signal,
                sample_rate=self.sample_rate,
                bands=self.bands,
                n_fft=self.n_fft,
                window=self.window,
            )
            rows.append([features[name] for name in feature_names])
        output = np.asarray(rows, dtype=float)
        return output[0] if single else output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        return {
            "sample_rate": self.sample_rate,
            "bands": self.bands,
            "n_fft": self.n_fft,
            "window": self.window,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.feature_names_ = None
        return self


__all__ = [
    "SignalHealthMonitor",
    "VibrationFeatureTransformer",
    "assess_signal_health",
    "signal_quality_report",
    "vibration_features",
    "windowed_vibration_features",
]
