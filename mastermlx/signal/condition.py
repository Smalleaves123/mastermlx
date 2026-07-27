"""Condition-monitoring helpers for sensor and vibration signals."""

from __future__ import annotations

import numpy as np

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


def vibration_features(signal, sample_rate, bands=None, n_fft=None, window="hann"):
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
        spectral_centroid(values, sample_rate=sample_rate, n_fft=n_fft, window=window),
        dtype=float,
    )
    bandwidth = np.asarray(
        spectral_bandwidth(values, sample_rate=sample_rate, n_fft=n_fft, window=window),
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


__all__ = ["VibrationFeatureTransformer", "signal_quality_report", "vibration_features"]
