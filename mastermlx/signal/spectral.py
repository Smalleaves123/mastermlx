"""Spectral density and cross-spectral analysis."""

from __future__ import annotations

import numpy as np

from .core import hamming_window, hann_window


def _as_signal(x, name="x"):
    arr = np.asarray(x)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    return arr.astype(complex if np.iscomplexobj(arr) else float, copy=False)


def _window(window, size):
    if window is None or (isinstance(window, str) and window == "boxcar"):
        out = np.ones(size, dtype=float)
    elif isinstance(window, str) and window == "hann":
        out = hann_window(size)
    elif isinstance(window, str) and window == "hamming":
        out = hamming_window(size)
    else:
        out = np.asarray(window, dtype=float)
        if out.ndim != 1 or out.size != size:
            raise ValueError("window must be a supported name or a 1D array matching nperseg")
    if not np.any(out):
        raise ValueError("window must not be all zeros")
    return out


def _detrend(segment, mode):
    if mode is None:
        return segment
    if mode == "constant":
        return segment - np.mean(segment)
    if mode != "linear":
        raise ValueError("detrend must be None, 'constant', or 'linear'")
    t = np.arange(segment.shape[1], dtype=float)
    t_mean = np.mean(t)
    dt = t - t_mean
    denom = np.sum(dt * dt)
    y_mean = np.mean(segment, axis=1, keepdims=True)
    slope = np.sum((segment - y_mean) * dt[None, :], axis=1, keepdims=True) / denom
    return segment - (y_mean + slope * dt[None, :])


def _segments(x, nperseg, noverlap, pad_end):
    n = x.size
    nperseg = min(int(nperseg), n)
    if nperseg < 1:
        raise ValueError("nperseg must be positive")
    noverlap = nperseg // 2 if noverlap is None else int(noverlap)
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")
    step = nperseg - noverlap
    starts = list(range(0, max(1, n - nperseg + 1), step))
    if pad_end and starts[-1] + nperseg < n:
        starts.append(starts[-1] + step)

    out = []
    for start in starts:
        segment = x[start : start + nperseg]
        if segment.size < nperseg:
            if not pad_end:
                continue
            segment = np.pad(segment, (0, nperseg - segment.size))
        out.append(segment)
    if not out:
        out.append(x[:nperseg])
    return np.asarray(out), nperseg, noverlap


def _spectra(x, y=None, sample_rate=1.0, nperseg=256, noverlap=None, nfft=None,
             window="hann", detrend="constant", scaling="density", pad_end=False):
    x = _as_signal(x, "x")
    y = None if y is None else _as_signal(y, "y")
    if y is not None and x.size != y.size:
        raise ValueError("x and y must have the same length")
    sample_rate = float(sample_rate)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if scaling not in {"density", "spectrum"}:
        raise ValueError("scaling must be 'density' or 'spectrum'")

    x_seg, nperseg, noverlap = _segments(x, nperseg, noverlap, pad_end)
    y_seg = None
    if y is not None:
        y_seg, _, _ = _segments(y, nperseg, noverlap, pad_end)
    nfft = nperseg if nfft is None else int(nfft)
    if nfft < nperseg:
        raise ValueError("nfft must be at least nperseg")
    win = _window(window, nperseg)
    norm = np.sum(win * win)
    real = not np.iscomplexobj(x) and (y is None or not np.iscomplexobj(y))

    x_seg = _detrend(x_seg, detrend) * win[None, :]
    xf = np.fft.rfft(x_seg, n=nfft, axis=1) if real else np.fft.fft(x_seg, n=nfft, axis=1)
    if y_seg is None:
        values = np.conj(xf) * xf
    else:
        y_seg = _detrend(y_seg, detrend) * win[None, :]
        yf = np.fft.rfft(y_seg, n=nfft, axis=1) if real else np.fft.fft(y_seg, n=nfft, axis=1)
        values = np.conj(xf) * yf

    values = values / (norm * sample_rate if scaling == "density" else norm)
    if real:
        if nfft % 2 == 0:
            values[:, 1:-1] *= 2.0
        else:
            values[:, 1:] *= 2.0
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    else:
        values = np.fft.fftshift(values, axes=1)
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
    return freqs, values


def _average(values, average):
    if average not in {"mean", "median"}:
        raise ValueError("average must be 'mean' or 'median'")
    if average == "mean":
        return np.mean(values, axis=0)
    if np.iscomplexobj(values):
        return np.median(values.real, axis=0) + 1j * np.median(values.imag, axis=0)
    return np.median(values, axis=0)


def welch_psd(x, sample_rate=1.0, nperseg=256, noverlap=None, nfft=None,
              window="hann", detrend="constant", scaling="density",
              average="mean", pad_end=False):
    """Estimate a one-sided or two-sided power spectral density with Welch's method."""

    freqs, values = _spectra(
        x, sample_rate=sample_rate, nperseg=nperseg, noverlap=noverlap,
        nfft=nfft, window=window, detrend=detrend, scaling=scaling,
        pad_end=pad_end,
    )
    return freqs, np.real(_average(values, average))


def cross_spectrum(x, y, sample_rate=1.0, nperseg=256, noverlap=None, nfft=None,
                   window="hann", detrend="constant", scaling="density",
                   average="mean", pad_end=False):
    """Estimate the cross power spectral density ``conj(X) * Y``."""

    freqs, values = _spectra(
        x, y=y, sample_rate=sample_rate, nperseg=nperseg, noverlap=noverlap,
        nfft=nfft, window=window, detrend=detrend, scaling=scaling,
        pad_end=pad_end,
    )
    return freqs, _average(values, average)


def coherence(x, y, sample_rate=1.0, nperseg=256, noverlap=None, nfft=None,
              window="hann", detrend="constant", scaling="density",
              average="mean", pad_end=False):
    """Estimate magnitude-squared coherence between two signals."""

    freqs, xx = _spectra(
        x, sample_rate=sample_rate, nperseg=nperseg, noverlap=noverlap,
        nfft=nfft, window=window, detrend=detrend, scaling=scaling,
        pad_end=pad_end,
    )
    _, yy = _spectra(
        y, sample_rate=sample_rate, nperseg=nperseg, noverlap=noverlap,
        nfft=nfft, window=window, detrend=detrend, scaling=scaling,
        pad_end=pad_end,
    )
    _, xy = _spectra(
        x, y=y, sample_rate=sample_rate, nperseg=nperseg, noverlap=noverlap,
        nfft=nfft, window=window, detrend=detrend, scaling=scaling,
        pad_end=pad_end,
    )
    pxx = np.real(_average(xx, average))
    pyy = np.real(_average(yy, average))
    pxy = _average(xy, average)
    value = np.divide(
        np.abs(pxy) ** 2,
        pxx * pyy,
        out=np.zeros_like(pxx, dtype=float),
        where=(pxx * pyy) > 1e-24,
    )
    return freqs, np.clip(value, 0.0, 1.0)


def periodogram(x, sample_rate=1.0, nfft=None, window="boxcar", detrend="constant",
                scaling="density"):
    """Estimate a signal's periodogram as a single-segment PSD."""

    return welch_psd(
        x, sample_rate=sample_rate, nperseg=np.asarray(x).size,
        noverlap=0, nfft=nfft, window=window, detrend=detrend,
        scaling=scaling, average="mean", pad_end=False,
    )


welch = welch_psd
cross_power_spectrum = cross_spectrum

__all__ = [
    "coherence",
    "cross_power_spectrum",
    "cross_spectrum",
    "periodogram",
    "welch",
    "welch_psd",
]
