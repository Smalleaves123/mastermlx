"""Analytic-signal and time-frequency analysis tools."""

from __future__ import annotations

import numpy as np

from ..accel.timefreq_ops import ridge_path


def _signal(x, name="x"):
    arr = np.asarray(x)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    return arr


def _real_signal(x, name="x"):
    arr = _signal(x, name)
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real-valued")
    return arr.astype(float, copy=False)


def analytic_signal(x, n_fft=None):
    """Return the analytic signal formed with an FFT Hilbert transform."""

    x = _real_signal(x)
    n_fft = x.size if n_fft is None else int(n_fft)
    if n_fft < x.size:
        raise ValueError("n_fft must be at least the signal length")
    if n_fft < 1:
        raise ValueError("n_fft must be positive")

    spectrum = np.fft.fft(x, n=n_fft)
    mask = np.zeros(n_fft, dtype=float)
    mask[0] = 1.0
    if n_fft % 2 == 0:
        mask[1 : n_fft // 2] = 2.0
        mask[n_fft // 2] = 1.0
    else:
        mask[1 : (n_fft + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * mask)[: x.size]


def hilbert_transform(x, n_fft=None):
    """Return the real Hilbert transform of a real-valued signal."""

    return np.imag(analytic_signal(x, n_fft=n_fft))


hilbert = hilbert_transform


def instantaneous_amplitude(x, n_fft=None):
    """Return the instantaneous amplitude, or envelope, of a signal."""

    return np.abs(analytic_signal(x, n_fft=n_fft))


def instantaneous_phase(x, n_fft=None, unwrap=True):
    """Return the instantaneous phase in radians."""

    phase = np.angle(analytic_signal(x, n_fft=n_fft))
    return np.unwrap(phase) if unwrap else phase


def instantaneous_frequency(x, sample_rate=1.0, n_fft=None, unwrap=True):
    """Return the instantaneous frequency in cycles per unit time."""

    sample_rate = float(sample_rate)
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    phase = instantaneous_phase(x, n_fft=n_fft, unwrap=unwrap)
    if phase.size < 2:
        raise ValueError("x must contain at least two samples")
    return np.gradient(phase) * sample_rate / (2.0 * np.pi)


def instantaneous_features(x, sample_rate=1.0, n_fft=None, unwrap=True):
    """Return analytic signal, amplitude, phase, and instantaneous frequency."""

    sample_rate = float(sample_rate)
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    analytic = analytic_signal(x, n_fft=n_fft)
    phase = np.angle(analytic)
    if unwrap:
        phase = np.unwrap(phase)
    if phase.size < 2:
        raise ValueError("x must contain at least two samples")
    return {
        "analytic": analytic,
        "amplitude": np.abs(analytic),
        "phase": phase,
        "frequency": np.gradient(phase) * sample_rate / (2.0 * np.pi),
    }


def _scales(scales):
    scales = np.asarray(scales, dtype=float)
    if scales.ndim != 1 or scales.size == 0:
        raise ValueError("scales must be a non-empty 1D array")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("scales must be finite and positive")
    return scales


def wavelet_scales(n_scales=32, min_scale=1.0, max_scale=None, spacing="log"):
    """Create linear or logarithmic CWT scales measured in samples."""

    n_scales = int(n_scales)
    min_scale = float(min_scale)
    if n_scales < 1:
        raise ValueError("n_scales must be positive")
    if min_scale <= 0.0:
        raise ValueError("min_scale must be positive")
    if max_scale is None:
        max_scale = min_scale * 2.0 ** max(0, n_scales - 1)
    max_scale = float(max_scale)
    if max_scale < min_scale:
        raise ValueError("max_scale must be at least min_scale")
    if spacing == "log":
        return np.geomspace(min_scale, max_scale, n_scales)
    if spacing == "linear":
        return np.linspace(min_scale, max_scale, n_scales)
    raise ValueError("spacing must be 'log' or 'linear'")


def _wavelet(scale, wavelet, w0):
    half = max(1, int(np.ceil(4.0 * scale)))
    time = np.arange(-half, half + 1, dtype=float) / scale
    gaussian = np.exp(-0.5 * time * time)
    if wavelet == "morlet":
        correction = np.exp(-0.5 * w0 * w0)
        values = np.pi ** -0.25 * (np.exp(1j * w0 * time) - correction) * gaussian
    elif wavelet in {"mexican_hat", "mexican-hat", "ricker"}:
        values = (2.0 / (np.sqrt(3.0) * np.pi ** 0.25)) * (1.0 - time * time) * gaussian
    else:
        raise ValueError("wavelet must be 'morlet' or 'mexican_hat'")
    return values / np.sqrt(scale)


def cwt(x, scales, sample_rate=1.0, wavelet="morlet", w0=6.0, pad_mode="reflect"):
    """Compute a continuous wavelet transform.

    Scales are measured in samples.  The returned pseudo-frequencies use the
    Morlet center-frequency mapping ``w0 * sample_rate / (2 pi * scale)``.
    Coefficients have shape ``(n_scales, n_samples)``.
    """

    x = _signal(x)
    scales = _scales(scales)
    sample_rate = float(sample_rate)
    w0 = float(w0)
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    if w0 <= 0.0:
        raise ValueError("w0 must be positive")
    if pad_mode not in {"reflect", "symmetric", "edge", "constant"}:
        raise ValueError("pad_mode must be reflect, symmetric, edge, or constant")

    coefficients = []
    for scale in scales:
        kernel = np.conj(_wavelet(scale, str(wavelet).lower(), w0)[::-1])
        half = (kernel.size - 1) // 2
        if pad_mode == "constant":
            padded = np.pad(x, (half, half), mode=pad_mode, constant_values=0.0)
        else:
            padded = np.pad(x, (half, half), mode=pad_mode)
        coefficients.append(np.convolve(padded, kernel, mode="valid"))
    coefficients = np.asarray(coefficients)
    frequencies = w0 * sample_rate / (2.0 * np.pi * scales)
    return scales, frequencies, coefficients


continuous_wavelet_transform = cwt
wavelet_transform = cwt


def wavelet_power(coefficients):
    """Return the wavelet power/scalogram from CWT coefficients."""

    coefficients = np.asarray(coefficients)
    if coefficients.ndim != 2 or 0 in coefficients.shape:
        raise ValueError("coefficients must be a non-empty 2D array")
    return np.abs(coefficients) ** 2


def extract_ridge(
    values,
    frequencies,
    smoothness=1.0,
    max_jump=None,
    log_power=True,
    times=None,
):
    """Extract a continuous maximum-energy ridge from a time-frequency map.

    ``values`` may be complex coefficients or a non-negative power map with
    shape ``(n_frequencies, n_times)``.  The result contains the selected
    frequency-bin indices, frequencies, power, time coordinates, and score.
    """

    values = np.asarray(values)
    frequencies = np.asarray(frequencies, dtype=float)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("values must be a non-empty 2D array")
    if frequencies.ndim != 1 or frequencies.size != values.shape[0]:
        raise ValueError("frequencies must match the first values dimension")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("frequencies must be finite")
    delta = np.diff(frequencies)
    if delta.size and not (np.all(delta > 0.0) or np.all(delta < 0.0)):
        raise ValueError("frequencies must be strictly monotonic")

    if np.iscomplexobj(values):
        power = np.abs(values) ** 2
    else:
        power = values.astype(float, copy=False)
        if np.any(power < 0.0):
            raise ValueError("real values must be non-negative power")
    smoothness = float(smoothness)
    if smoothness < 0.0:
        raise ValueError("smoothness must be non-negative")
    if max_jump is not None:
        max_jump = int(max_jump)
        if max_jump < 0:
            raise ValueError("max_jump must be non-negative")
    if times is None:
        times = np.arange(values.shape[1], dtype=float)
    else:
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size != values.shape[1]:
            raise ValueError("times must match the second values dimension")

    score = np.log(np.maximum(power, 1e-24)) if log_power else power.copy()
    n_freqs, n_times = score.shape
    indices = ridge_path(score, smoothness, max_jump)
    columns = np.arange(n_times)
    path_score = float(score[indices[0], 0])
    if n_times > 1:
        jumps = np.diff(indices)
        path_score += float(
            np.sum(score[indices[1:], np.arange(1, n_times)] - smoothness * jumps * jumps)
        )
    return {
        "indices": indices,
        "frequencies": frequencies[indices],
        "power": power[indices, columns],
        "time": times,
        "score": path_score,
    }


time_frequency_ridge = extract_ridge
ridge = extract_ridge


__all__ = [
    "analytic_signal",
    "continuous_wavelet_transform",
    "cwt",
    "extract_ridge",
    "hilbert",
    "hilbert_transform",
    "instantaneous_amplitude",
    "instantaneous_features",
    "instantaneous_frequency",
    "instantaneous_phase",
    "ridge",
    "time_frequency_ridge",
    "wavelet_power",
    "wavelet_scales",
    "wavelet_transform",
]
