from __future__ import annotations

import numpy as np

from .core import convolve1d


def normalize_signal(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / (std + float(eps))


def pre_emphasis(x, coef=0.97):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    if not 0.0 <= coef <= 1.0:
        raise ValueError("coef must be in [0, 1]")
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    y[1:] = x[1:] - coef * x[:-1]
    return y


def de_emphasis(x, coef=0.97):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    if not 0.0 <= coef <= 1.0:
        raise ValueError("coef must be in [0, 1]")
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, x.size):
        y[i] = x[i] + coef * y[i - 1]
    return y


def _validate_1d_signal(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    return x


def _validate_fir_params(low_cutoff, high_cutoff, sample_rate, num_taps):
    low_cutoff = float(low_cutoff)
    high_cutoff = float(high_cutoff)
    sample_rate = float(sample_rate)
    num_taps = int(num_taps)
    if num_taps < 3 or num_taps % 2 == 0:
        raise ValueError("num_taps must be an odd integer at least 3")
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    nyquist = sample_rate / 2.0
    if not (0.0 <= low_cutoff < high_cutoff <= nyquist):
        raise ValueError("cutoffs must satisfy 0 <= low_cutoff < high_cutoff <= sample_rate / 2")
    if low_cutoff == 0.0 and high_cutoff == nyquist:
        raise ValueError("band edges must define a non-trivial filter")
    return low_cutoff, high_cutoff, sample_rate, num_taps, nyquist


def _resolve_window(window, num_taps):
    if window == "hann":
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(num_taps) / (num_taps - 1))
    if window == "hamming":
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(num_taps) / (num_taps - 1))
    if window is None:
        return np.ones(num_taps, dtype=float)
    raise ValueError("window must be one of: hann, hamming, None")


def _design_lowpass_kernel(cutoff, sample_rate, num_taps):
    idx = np.arange(num_taps) - (num_taps - 1) / 2.0
    kernel = 2.0 * cutoff / sample_rate * np.sinc(2.0 * cutoff * idx / sample_rate)
    return kernel


def _design_bandpass_kernel(low_cutoff, high_cutoff, sample_rate, num_taps):
    high = _design_lowpass_kernel(high_cutoff, sample_rate, num_taps)
    low = _design_lowpass_kernel(low_cutoff, sample_rate, num_taps)
    return high - low


def _design_bandstop_kernel(low_cutoff, high_cutoff, sample_rate, num_taps):
    low = _design_lowpass_kernel(low_cutoff, sample_rate, num_taps)
    high = _design_lowpass_kernel(high_cutoff, sample_rate, num_taps)
    kernel = np.zeros(num_taps, dtype=float)
    kernel[(num_taps - 1) // 2] = 1.0
    return kernel + low - high


def _apply_fir(x, kernel, zero_mean=False, normalize=None):
    kernel = np.asarray(kernel, dtype=float)
    if zero_mean:
        kernel = kernel - np.mean(kernel)
    if normalize == "l1":
        kernel = kernel / (np.sum(np.abs(kernel)) + 1e-12)
    elif normalize == "sum":
        denom = np.sum(kernel)
        if abs(denom) < 1e-12:
            raise ValueError("kernel sum is too close to zero to normalize")
        kernel = kernel / denom
    return convolve1d(x, kernel, mode="same")


def bandpass_filter(x, low_cutoff, high_cutoff, sample_rate, num_taps=101, window="hann"):
    x = _validate_1d_signal(x)
    low_cutoff, high_cutoff, sample_rate, num_taps, nyquist = _validate_fir_params(low_cutoff, high_cutoff, sample_rate, num_taps)
    if not (0.0 < low_cutoff < high_cutoff < nyquist):
        raise ValueError("cutoffs must satisfy 0 < low_cutoff < high_cutoff < sample_rate / 2")
    kernel = _design_bandpass_kernel(low_cutoff, high_cutoff, sample_rate, num_taps)
    kernel *= _resolve_window(window, num_taps)
    return _apply_fir(x, kernel, zero_mean=True, normalize="l1")


def bandstop_filter(x, low_cutoff, high_cutoff, sample_rate, num_taps=101, window="hann"):
    x = _validate_1d_signal(x)
    low_cutoff, high_cutoff, sample_rate, num_taps, nyquist = _validate_fir_params(low_cutoff, high_cutoff, sample_rate, num_taps)
    if low_cutoff == 0.0 and high_cutoff == nyquist:
        return np.zeros_like(x, dtype=float)

    kernel = _design_bandstop_kernel(low_cutoff, high_cutoff, sample_rate, num_taps)
    kernel *= _resolve_window(window, num_taps)
    return _apply_fir(x, kernel, normalize="sum")


def notch_filter(x, frequency, bandwidth, sample_rate, num_taps=101, window="hann"):
    frequency = float(frequency)
    bandwidth = float(bandwidth)
    sample_rate = float(sample_rate)
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    low_cutoff = max(0.0, frequency - bandwidth / 2.0)
    high_cutoff = min(sample_rate / 2.0, frequency + bandwidth / 2.0)
    if not low_cutoff < high_cutoff:
        raise ValueError("frequency and bandwidth define an empty notch")
    return bandstop_filter(
        x,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        sample_rate=sample_rate,
        num_taps=num_taps,
        window=window,
    )


norm_sig = normalize_signal
pre_emph = pre_emphasis
de_emph = de_emphasis
