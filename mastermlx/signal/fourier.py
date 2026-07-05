from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from .core import hamming_window, hann_window, stft


def _as_1d_signal(x, name="x"):
    arr = np.asarray(x)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    return arr


def _resolve_window(window, n_samples):
    if window is None:
        return np.ones(int(n_samples), dtype=float)
    if window == "hann":
        return hann_window(n_samples)
    if window == "hamming":
        return hamming_window(n_samples)
    raise ValueError("window must be one of: hann, hamming, None")


def frequency_bins(n_fft, sample_rate=1.0, real=True):
    n_fft = int(n_fft)
    sample_rate = float(sample_rate)
    if n_fft < 1:
        raise ValueError("n_fft must be at least 1")
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    if real:
        return np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return np.fft.fftfreq(n_fft, d=1.0 / sample_rate)


def fft_signal(x, n_fft=None):
    x = _as_1d_signal(x)
    n_fft = x.size if n_fft is None else int(n_fft)
    if n_fft < 1:
        raise ValueError("n_fft must be at least 1")
    return np.fft.fft(np.asarray(x, dtype=complex), n=n_fft)


def ifft_signal(spectrum, n_fft=None, length=None):
    spec = np.asarray(spectrum, dtype=complex)
    if spec.ndim != 1 or spec.size == 0:
        raise ValueError("spectrum must be a non-empty 1D array")
    n_fft = spec.size if n_fft is None else int(n_fft)
    if n_fft < 1:
        raise ValueError("n_fft must be at least 1")
    x = np.fft.ifft(spec, n=n_fft)
    if length is not None:
        x = x[: int(length)]
    return x


def rfft_signal(x, n_fft=None):
    x = _as_1d_signal(x)
    n_fft = x.size if n_fft is None else int(n_fft)
    if n_fft < 1:
        raise ValueError("n_fft must be at least 1")
    return np.fft.rfft(np.asarray(x, dtype=float), n=n_fft)


def irfft_signal(spectrum, n_fft=None, length=None):
    spec = np.asarray(spectrum, dtype=complex)
    if spec.ndim != 1 or spec.size == 0:
        raise ValueError("spectrum must be a non-empty 1D array")
    if n_fft is None:
        n_fft = (spec.size - 1) * 2
    n_fft = int(n_fft)
    if n_fft < 1:
        raise ValueError("n_fft must be at least 1")
    x = np.fft.irfft(spec, n=n_fft)
    if length is not None:
        x = x[: int(length)]
    return x


def fft_spectrum(x, sample_rate=1.0, n_fft=None, window="hann"):
    x = _as_1d_signal(x)
    n_fft = x.size if n_fft is None else int(n_fft)
    win = _resolve_window(window, x.size)
    if win.size != x.size:
        raise ValueError("window size must match signal length")
    spectrum = np.fft.rfft(np.asarray(x, dtype=float) * win, n=n_fft)
    freqs = frequency_bins(n_fft, sample_rate=sample_rate, real=True)
    return freqs, spectrum


def inverse_fft_spectrum(spectrum, n_fft=None, length=None):
    return irfft_signal(spectrum, n_fft=n_fft, length=length)


def amplitude_spectrum(x, sample_rate=1.0, n_fft=None, window="hann", normalize=False):
    freqs, spectrum = fft_spectrum(x, sample_rate=sample_rate, n_fft=n_fft, window=window)
    amplitude = np.abs(spectrum)
    if normalize:
        amplitude = amplitude / (np.max(amplitude) + 1e-12)
    return freqs, amplitude


def power_spectrum(x, sample_rate=1.0, n_fft=None, window="hann", normalize=False):
    freqs, spectrum = fft_spectrum(x, sample_rate=sample_rate, n_fft=n_fft, window=window)
    power = np.abs(spectrum) ** 2
    if normalize:
        power = power / (np.max(power) + 1e-12)
    return freqs, power


def phase_spectrum(x, sample_rate=1.0, n_fft=None, window="hann", unwrap=False):
    freqs, spectrum = fft_spectrum(x, sample_rate=sample_rate, n_fft=n_fft, window=window)
    phase = np.angle(spectrum)
    if unwrap:
        phase = np.unwrap(phase)
    return freqs, phase


def dominant_frequency(x, sample_rate=1.0, n_fft=None, window="hann", ignore_dc=True):
    freqs, amplitude = amplitude_spectrum(x, sample_rate=sample_rate, n_fft=n_fft, window=window)
    if amplitude.size == 0:
        raise ValueError("signal must produce a non-empty spectrum")
    start = 1 if ignore_dc and amplitude.size > 1 else 0
    idx = start + int(np.argmax(amplitude[start:]))
    return float(freqs[idx]), float(amplitude[idx]), int(idx)


def top_frequency_peaks(x, sample_rate=1.0, n_fft=None, window="hann", top_k=5, ignore_dc=True):
    freqs, amplitude = amplitude_spectrum(x, sample_rate=sample_rate, n_fft=n_fft, window=window)
    if amplitude.size == 0:
        return np.empty((0, 2), dtype=float)
    top_k = int(top_k)
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if top_k == 0:
        return np.empty((0, 2), dtype=float)
    start = 1 if ignore_dc and amplitude.size > 1 else 0
    idx = np.argsort(amplitude[start:])[::-1][:top_k] + start
    peaks = np.column_stack([freqs[idx], amplitude[idx]])
    return peaks


def band_energy(x, sample_rate=1.0, bands=None, n_fft=None, window="hann", normalize=False):
    sample_rate = float(sample_rate)
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    if bands is None:
        bands = [(0.0, sample_rate / 4.0), (sample_rate / 4.0, sample_rate / 2.0)]
    bands = [(float(lo), float(hi)) for lo, hi in bands]
    freqs, power = power_spectrum(x, sample_rate=sample_rate, n_fft=n_fft, window=window, normalize=False)
    energies = []
    total = np.sum(power) + 1e-12
    for low, high in bands:
        if low < 0.0 or high <= low:
            raise ValueError("each band must satisfy 0 <= low < high")
        mask = (freqs >= low) & (freqs < high)
        energy = float(np.sum(power[mask]))
        if normalize:
            energy /= total
        energies.append(energy)
    return np.asarray(energies, dtype=float)


def stft_spectrum(x, frame_length=256, hop_length=None, window="hann", n_fft=None, pad_end=True, reduce=None):
    x = _as_1d_signal(x)
    frame_length = int(frame_length)
    hop_length = frame_length // 2 if hop_length is None else int(hop_length)
    n_fft = frame_length if n_fft is None else int(n_fft)
    if frame_length < 1:
        raise ValueError("frame_length must be at least 1")
    if hop_length < 1:
        raise ValueError("hop_length must be at least 1")
    if n_fft < frame_length:
        raise ValueError("n_fft must be at least frame_length")

    spectrum = stft(x, frame_length=frame_length, hop_length=hop_length, window=window, n_fft=n_fft, pad_end=pad_end)
    if spectrum.size == 0:
        empty = np.empty((0, n_fft // 2 + 1), dtype=float)
        if reduce is None:
            return empty
        return np.empty(0, dtype=float)

    magnitude = np.abs(spectrum)
    if reduce is None:
        return magnitude

    if isinstance(reduce, str):
        reducers = (reduce,)
    else:
        reducers = tuple(reduce)

    pieces = []
    for name in reducers:
        if name == "mean":
            pieces.append(np.mean(magnitude, axis=0))
        elif name == "std":
            pieces.append(np.std(magnitude, axis=0))
        elif name == "max":
            pieces.append(np.max(magnitude, axis=0))
        elif name == "median":
            pieces.append(np.median(magnitude, axis=0))
        elif name == "min":
            pieces.append(np.min(magnitude, axis=0))
        else:
            raise ValueError("reduce must contain only: mean, std, max, median, min")
    return np.concatenate([np.asarray(piece, dtype=float).ravel() for piece in pieces])


def spectral_feature_vector(
    x,
    sample_rate=1.0,
    frame_length=256,
    hop_length=None,
    n_fft=None,
    window="hann",
    pad_end=True,
    fft_output="amplitude",
    stft_reduce=("mean", "std"),
    include_fft=True,
    include_stft=True,
    normalize=False,
):
    x = _as_1d_signal(x)
    pieces = []

    if include_fft:
        if fft_output == "amplitude":
            _, feat = amplitude_spectrum(x, sample_rate=sample_rate, n_fft=n_fft, window=window, normalize=normalize)
        elif fft_output == "power":
            _, feat = power_spectrum(x, sample_rate=sample_rate, n_fft=n_fft, window=window, normalize=normalize)
        elif fft_output == "phase":
            _, feat = phase_spectrum(x, sample_rate=sample_rate, n_fft=n_fft, window=window, unwrap=False)
        elif fft_output == "complex":
            _, feat = fft_spectrum(x, sample_rate=sample_rate, n_fft=n_fft, window=window)
        else:
            raise ValueError("fft_output must be one of: amplitude, power, phase, complex")
        pieces.append(np.asarray(feat).ravel())

    if include_stft:
        feat = stft_spectrum(
            x,
            frame_length=frame_length,
            hop_length=hop_length,
            window=window,
            n_fft=n_fft,
            pad_end=pad_end,
            reduce=stft_reduce,
        )
        pieces.append(np.asarray(feat).ravel())

    if not pieces:
        return np.empty(0, dtype=float)
    return np.concatenate([np.asarray(piece, dtype=float).ravel() for piece in pieces])


class FourierTransformer(BaseTransformer):
    """Transform raw 1D signals into Fourier-domain features."""

    def __init__(self, sample_rate=1.0, n_fft=None, window="hann", output="amplitude", normalize=False, ignore_dc=True, top_k=5, bands=None):
        self.sample_rate = float(sample_rate)
        self.n_fft = None if n_fft is None else int(n_fft)
        self.window = window
        self.output = output
        self.normalize = bool(normalize)
        self.ignore_dc = bool(ignore_dc)
        self.top_k = int(top_k)
        self.bands = bands

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = _as_1d_signal(X)
        if self.output == "amplitude":
            _, spec = amplitude_spectrum(x, sample_rate=self.sample_rate, n_fft=self.n_fft, window=self.window, normalize=self.normalize)
            return spec
        if self.output == "power":
            _, spec = power_spectrum(x, sample_rate=self.sample_rate, n_fft=self.n_fft, window=self.window, normalize=self.normalize)
            return spec
        if self.output == "phase":
            _, spec = phase_spectrum(x, sample_rate=self.sample_rate, n_fft=self.n_fft, window=self.window, unwrap=False)
            return spec
        if self.output == "complex":
            _, spec = fft_spectrum(x, sample_rate=self.sample_rate, n_fft=self.n_fft, window=self.window)
            return spec
        if self.output == "dominant":
            freq, amplitude, idx = dominant_frequency(x, sample_rate=self.sample_rate, n_fft=self.n_fft, window=self.window, ignore_dc=self.ignore_dc)
            return np.array([freq, amplitude, idx], dtype=float)
        if self.output == "peaks":
            return top_frequency_peaks(
                x,
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                window=self.window,
                top_k=self.top_k,
                ignore_dc=self.ignore_dc,
            )
        if self.output == "band_energy":
            return band_energy(x, sample_rate=self.sample_rate, bands=self.bands, n_fft=self.n_fft, window=self.window, normalize=self.normalize)
        raise ValueError("output must be one of: amplitude, power, phase, complex, dominant, peaks, band_energy")


class SpectralFeatureTransformer(BaseTransformer):
    """Build a fixed-width feature vector from FFT and STFT summaries."""

    def __init__(
        self,
        sample_rate=1.0,
        frame_length=256,
        hop_length=None,
        n_fft=None,
        window="hann",
        pad_end=True,
        fft_output="amplitude",
        stft_reduce=("mean", "std"),
        include_fft=True,
        include_stft=True,
        normalize=False,
    ):
        self.sample_rate = float(sample_rate)
        self.frame_length = int(frame_length)
        self.hop_length = None if hop_length is None else int(hop_length)
        self.n_fft = None if n_fft is None else int(n_fft)
        self.window = window
        self.pad_end = bool(pad_end)
        self.fft_output = fft_output
        self.stft_reduce = stft_reduce
        self.include_fft = bool(include_fft)
        self.include_stft = bool(include_stft)
        self.normalize = bool(normalize)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = _as_1d_signal(X)
        return spectral_feature_vector(
            x,
            sample_rate=self.sample_rate,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            window=self.window,
            pad_end=self.pad_end,
            fft_output=self.fft_output,
            stft_reduce=self.stft_reduce,
            include_fft=self.include_fft,
            include_stft=self.include_stft,
            normalize=self.normalize,
        )


class InverseFourierTransformer(BaseTransformer):
    """Reconstruct a signal from a Fourier spectrum."""

    def __init__(self, n_fft=None, length=None, real=True):
        self.n_fft = None if n_fft is None else int(n_fft)
        self.length = length
        self.real = bool(real)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.real:
            return irfft_signal(X, n_fft=self.n_fft, length=self.length)
        return ifft_signal(X, n_fft=self.n_fft, length=self.length)


__all__ = [
    "FourierTransformer",
    "InverseFourierTransformer",
    "amplitude_spectrum",
    "band_energy",
    "dominant_frequency",
    "fft_signal",
    "fft_spectrum",
    "frequency_bins",
    "inverse_fft_spectrum",
    "ifft_signal",
    "irfft_signal",
    "phase_spectrum",
    "power_spectrum",
    "rfft_signal",
    "SpectralFeatureTransformer",
    "spectral_feature_vector",
    "stft_spectrum",
    "top_frequency_peaks",
]
