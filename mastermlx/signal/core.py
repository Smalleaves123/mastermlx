from __future__ import annotations

import numpy as np

from ..accel.signal_ops import frame_signal as _frame_signal

def hamming_window(n):
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    if n == 1:
        return np.ones(1, dtype=float)
    idx = np.arange(n, dtype=float)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * idx / (n - 1))


def hann_window(n):
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    if n == 1:
        return np.ones(1, dtype=float)
    idx = np.arange(n, dtype=float)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * idx / (n - 1))


def convolve1d(x, kernel, mode="full"):
    x = np.asarray(x, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if x.ndim != 1 or kernel.ndim != 1:
        raise ValueError("x and kernel must be 1D arrays")
    if x.size == 0 or kernel.size == 0:
        raise ValueError("x and kernel must be non-empty")
    if mode not in {"full", "same", "valid"}:
        raise ValueError("mode must be one of: full, same, valid")
    return np.convolve(x, kernel, mode=mode)


def moving_average(x, window_size, mode="valid"):
    window_size = int(window_size)
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    kernel = np.ones(window_size, dtype=float) / window_size
    return convolve1d(x, kernel, mode=mode)


def autocorrelation(x, max_lag=None, normalize=True):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    x = x - np.mean(x)
    full = np.correlate(x, x, mode="full")
    mid = x.size - 1
    if max_lag is None:
        max_lag = x.size - 1
    max_lag = int(max_lag)
    if max_lag < 0 or max_lag >= x.size:
        raise ValueError("max_lag must be in [0, len(x) - 1]")
    corr = full[mid : mid + max_lag + 1]
    if normalize and corr[0] != 0:
        corr = corr / corr[0]
    return corr


def frame_signal(x, frame_length, hop_length, pad_end=False):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    frame_length = int(frame_length)
    hop_length = int(hop_length)
    if frame_length < 1 or hop_length < 1:
        raise ValueError("frame_length and hop_length must be at least 1")

    return _frame_signal(x, frame_length, hop_length, pad_end=pad_end)


def stft(x, frame_length=256, hop_length=None, window="hann", n_fft=None, pad_end=True):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    if hop_length is None:
        hop_length = frame_length // 2
    if n_fft is None:
        n_fft = frame_length
    if n_fft < frame_length:
        raise ValueError("n_fft must be at least frame_length")

    if window == "hann":
        win = hann_window(frame_length)
    elif window == "hamming":
        win = hamming_window(frame_length)
    elif window is None:
        win = np.ones(frame_length, dtype=float)
    else:
        raise ValueError("window must be one of: hann, hamming, None")

    frames = frame_signal(x, frame_length, hop_length, pad_end=pad_end)
    if frames.size == 0:
        return np.empty((0, n_fft // 2 + 1), dtype=complex)
    windowed = frames * win[None, :]
    return np.fft.rfft(windowed, n=n_fft, axis=1)


def istft(spectrogram, frame_length=256, hop_length=None, window="hann", length=None):
    spec = np.asarray(spectrogram)
    if spec.ndim != 2:
        raise ValueError("spectrogram must be a 2D array")
    if hop_length is None:
        hop_length = frame_length // 2
    if window == "hann":
        win = hann_window(frame_length)
    elif window == "hamming":
        win = hamming_window(frame_length)
    elif window is None:
        win = np.ones(frame_length, dtype=float)
    else:
        raise ValueError("window must be one of: hann, hamming, None")

    frames = np.fft.irfft(spec, n=frame_length, axis=1)
    x = overlap_add(frames, hop_length=hop_length, window=win)
    if length is not None:
        x = x[: int(length)]
    return x


def overlap_add(frames, hop_length, window=None, length=None):
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 2:
        raise ValueError("frames must be a 2D array")
    hop_length = int(hop_length)
    if hop_length < 1:
        raise ValueError("hop_length must be at least 1")
    if window is not None:
        window = np.asarray(window, dtype=float)
        if window.ndim != 1 or window.size != frames.shape[1]:
            raise ValueError("window must be a 1D array matching frame length")
        frames = frames * window[None, :]

    n_frames, frame_length = frames.shape
    if n_frames == 0:
        return np.empty(0, dtype=float)
    n_samples = frame_length + (n_frames - 1) * hop_length
    x = np.zeros(n_samples, dtype=float)
    norm = np.zeros(n_samples, dtype=float)
    if window is None:
        win = np.ones(frame_length, dtype=float)
    else:
        win = np.asarray(window, dtype=float)
    for i in range(n_frames):
        start = i * hop_length
        x[start : start + frame_length] += frames[i]
        norm[start : start + frame_length] += win**2
    mask = norm > 0
    x[mask] /= norm[mask]
    if length is not None:
        x = x[: int(length)]
    return x
