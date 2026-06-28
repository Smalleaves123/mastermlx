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


def bandpass_filter(x, low_cutoff, high_cutoff, sample_rate, num_taps=101, window="hann"):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    low_cutoff = float(low_cutoff)
    high_cutoff = float(high_cutoff)
    sample_rate = float(sample_rate)
    num_taps = int(num_taps)
    if num_taps < 3 or num_taps % 2 == 0:
        raise ValueError("num_taps must be an odd integer at least 3")
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    nyquist = sample_rate / 2.0
    if not (0.0 < low_cutoff < high_cutoff < nyquist):
        raise ValueError("cutoffs must satisfy 0 < low_cutoff < high_cutoff < sample_rate / 2")

    idx = np.arange(num_taps) - (num_taps - 1) / 2.0
    high = 2.0 * high_cutoff / sample_rate * np.sinc(2.0 * high_cutoff * idx / sample_rate)
    low = 2.0 * low_cutoff / sample_rate * np.sinc(2.0 * low_cutoff * idx / sample_rate)
    kernel = high - low

    if window == "hann":
        win = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(num_taps) / (num_taps - 1))
    elif window == "hamming":
        win = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(num_taps) / (num_taps - 1))
    elif window is None:
        win = np.ones(num_taps, dtype=float)
    else:
        raise ValueError("window must be one of: hann, hamming, None")

    kernel *= win
    kernel -= np.mean(kernel)
    kernel /= np.sum(np.abs(kernel)) + 1e-12
    return convolve1d(x, kernel, mode="same")


norm_sig = normalize_signal
pre_emph = pre_emphasis
de_emph = de_emphasis
