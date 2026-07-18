"""Cepstral, envelope, and cyclostationary signal analysis."""

from __future__ import annotations

import numpy as np

from .spectral import _detrend, _segments, _window
from .systems import butterworth, zero_phase_filter
from .time_frequency import instantaneous_amplitude


def _real_signal(x, name="x"):
    arr = np.asarray(x)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real-valued")
    return arr.astype(float, copy=False)


def _fft_size(n_samples, n_fft):
    n_fft = n_samples if n_fft is None else int(n_fft)
    if n_fft < n_samples:
        raise ValueError("n_fft must be at least the signal length")
    return n_fft


def real_cepstrum(x, sample_rate=1.0, n_fft=None, log_base="natural"):
    """Return real cepstrum values and quefrency in seconds."""

    x = _real_signal(x)
    sample_rate = float(sample_rate)
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    n_fft = _fft_size(x.size, n_fft)
    spectrum = np.fft.rfft(x, n=n_fft)
    log_magnitude = np.log(np.maximum(np.abs(spectrum), 1e-12))
    if log_base == "10":
        log_magnitude /= np.log(10.0)
    elif log_base != "natural":
        raise ValueError("log_base must be 'natural' or '10'")
    cepstrum = np.fft.irfft(log_magnitude, n=n_fft)
    quefrency = np.arange(n_fft, dtype=float) / sample_rate
    return quefrency, cepstrum


def complex_cepstrum(x, sample_rate=1.0, n_fft=None, log_base="natural"):
    """Return the complex cepstrum and quefrency in seconds."""

    x = _real_signal(x)
    sample_rate = float(sample_rate)
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    n_fft = _fft_size(x.size, n_fft)
    spectrum = np.fft.fft(x, n=n_fft)
    phase = np.unwrap(np.angle(spectrum))
    log_magnitude = np.log(np.maximum(np.abs(spectrum), 1e-12))
    if log_base == "10":
        log_magnitude /= np.log(10.0)
    elif log_base != "natural":
        raise ValueError("log_base must be 'natural' or '10'")
    cepstrum = np.fft.ifft(log_magnitude + 1j * phase)
    quefrency = np.arange(n_fft, dtype=float) / sample_rate
    return quefrency, cepstrum


def cepstrum_peaks(
    quefrency,
    values,
    n_peaks=5,
    min_quefrency=0.0,
    max_quefrency=None,
    min_distance=1,
):
    """Return local real-cepstrum peaks as ``(quefrency, value)`` rows."""

    quefrency = np.asarray(quefrency, dtype=float)
    values = np.asarray(values)
    if quefrency.ndim != 1 or values.ndim != 1 or quefrency.size != values.size or not quefrency.size:
        raise ValueError("quefrency and values must be matching non-empty 1D arrays")
    if not np.all(np.isfinite(quefrency)) or np.any(np.diff(quefrency) <= 0.0):
        raise ValueError("quefrency must be finite and strictly increasing")
    n_peaks = int(n_peaks)
    min_distance = int(min_distance)
    min_quefrency = float(min_quefrency)
    max_quefrency = np.inf if max_quefrency is None else float(max_quefrency)
    if n_peaks < 0 or min_distance < 1:
        raise ValueError("n_peaks must be non-negative and min_distance must be positive")
    if min_quefrency < 0.0 or max_quefrency <= min_quefrency:
        raise ValueError("quefrency bounds must be increasing and non-negative")
    score = np.real(values) if np.iscomplexobj(values) else values.astype(float, copy=False)
    candidates = []
    for idx in range(1, values.size - 1):
        if not min_quefrency <= quefrency[idx] <= max_quefrency:
            continue
        if score[idx] >= score[idx - 1] and score[idx] > score[idx + 1]:
            candidates.append(idx)
    candidates.sort(key=lambda idx: score[idx], reverse=True)
    selected: list[int] = []
    for idx in candidates:
        if all(abs(idx - other) >= min_distance for other in selected):
            selected.append(idx)
            if len(selected) == n_peaks:
                break
    selected.sort(key=lambda idx: quefrency[idx])
    return np.column_stack([quefrency[selected], values[selected]]) if selected else np.empty((0, 2), dtype=values.dtype)


def envelope_demodulate(x, sample_rate=1.0, carrier_band=None, order=4):
    """Extract a signal envelope, optionally after bandpass demodulation."""

    x = _real_signal(x)
    sample_rate = float(sample_rate)
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    filtered = x
    if carrier_band is not None:
        band = np.asarray(carrier_band, dtype=float).ravel()
        if band.size != 2 or not 0.0 < band[0] < band[1] < sample_rate / 2.0:
            raise ValueError("carrier_band must contain two frequencies below Nyquist")
        b, a = butterworth(order, band, sample_rate=sample_rate, btype="bandpass")
        filtered = zero_phase_filter(x, b, a)
    return instantaneous_amplitude(filtered)


envelope = envelope_demodulate
envelope_demodulation = envelope_demodulate


def envelope_spectrum(
    x,
    sample_rate=1.0,
    n_fft=None,
    carrier_band=None,
    order=4,
    window="hann",
    remove_mean=True,
):
    """Return the one-sided amplitude spectrum of a demodulated envelope."""

    x = _real_signal(x)
    sample_rate = float(sample_rate)
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    n_fft = _fft_size(x.size, n_fft)
    env = envelope_demodulate(x, sample_rate, carrier_band=carrier_band, order=order)
    if remove_mean:
        env = env - np.mean(env)
    win = _window(window, x.size)
    spectrum = np.abs(np.fft.rfft(env * win, n=n_fft)) / np.sum(win)
    if n_fft % 2 == 0:
        spectrum[1:-1] *= 2.0
    else:
        spectrum[1:] *= 2.0
    return np.fft.rfftfreq(n_fft, d=1.0 / sample_rate), spectrum


def _interp_complex(freq, values, points):
    return np.interp(points, freq, values.real, left=0.0, right=0.0) + 1j * np.interp(
        points, freq, values.imag, left=0.0, right=0.0
    )


def cyclic_spectrum(
    x,
    cyclic_frequencies=None,
    sample_rate=1.0,
    nperseg=256,
    noverlap=None,
    nfft=None,
    window="hann",
    detrend="constant",
):
    """Estimate the spectral correlation density ``S_x^alpha(f)``.

    Frequencies and cyclic frequencies are expressed in Hz.  The returned
    array has shape ``(n_cyclic_frequencies, n_frequencies)`` and uses a
    two-sided frequency axis.
    """

    x = np.asarray(x)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    sample_rate = float(sample_rate)
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    if cyclic_frequencies is None:
        cyclic_frequencies = [0.0]
    cyclic_frequencies = np.asarray(cyclic_frequencies, dtype=float)
    if cyclic_frequencies.ndim != 1 or cyclic_frequencies.size == 0 or not np.all(np.isfinite(cyclic_frequencies)):
        raise ValueError("cyclic_frequencies must be a non-empty finite 1D array")

    segments, nperseg, _ = _segments(x, nperseg, noverlap, pad_end=False)
    nfft = nperseg if nfft is None else int(nfft)
    if nfft < nperseg:
        raise ValueError("nfft must be at least nperseg")
    win = _window(window, nperseg)
    norm = np.sum(win * win) * sample_rate
    frequencies = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
    estimate = np.zeros((cyclic_frequencies.size, nfft), dtype=complex)
    for segment in segments:
        segment = _detrend(segment[None, :], detrend)[0] * win
        spectrum = np.fft.fftshift(np.fft.fft(segment, n=nfft))
        for row, alpha in enumerate(cyclic_frequencies):
            plus = _interp_complex(frequencies, spectrum, frequencies + alpha / 2.0)
            minus = _interp_complex(frequencies, spectrum, frequencies - alpha / 2.0)
            estimate[row] += plus * np.conj(minus) / norm
    estimate /= segments.shape[0]
    return frequencies, cyclic_frequencies, estimate


cyclostationary_spectrum = cyclic_spectrum
cyclic_spectral_density = cyclic_spectrum


__all__ = [
    "cepstrum_peaks",
    "complex_cepstrum",
    "cyclic_spectral_density",
    "cyclic_spectrum",
    "cyclostationary_spectrum",
    "envelope",
    "envelope_demodulate",
    "envelope_demodulation",
    "envelope_spectrum",
    "real_cepstrum",
]
