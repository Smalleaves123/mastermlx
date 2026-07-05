from __future__ import annotations

import numpy as np


def hz_to_mel(freq):
    freq = np.asarray(freq, dtype=float)
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_to_hz(mel):
    mel = np.asarray(mel, dtype=float)
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filter_bank(sample_rate, n_fft, n_mels=26, fmin=0.0, fmax=None):
    sample_rate = float(sample_rate)
    n_fft = int(n_fft)
    n_mels = int(n_mels)
    fmin = float(fmin)
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    if n_fft < 2:
        raise ValueError("n_fft must be at least 2")
    if n_mels < 1:
        raise ValueError("n_mels must be at least 1")
    if fmin < 0.0:
        raise ValueError("fmin must be non-negative")
    if fmax is None:
        fmax = sample_rate / 2.0
    fmax = float(fmax)
    if not fmin < fmax <= sample_rate / 2.0 + 1e-12:
        raise ValueError("fmax must be in (fmin, sample_rate / 2]")

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft // 2 + 1) * hz_points / (sample_rate / 2.0)).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    bank = np.zeros((n_mels, n_fft // 2 + 1), dtype=float)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center == left:
            center = min(center + 1, n_fft // 2)
        if right == center:
            right = min(right + 1, n_fft // 2 + 1)
        if left < center:
            bank[m - 1, left:center] = (np.arange(left, center) - left) / max(center - left, 1)
        if center < right:
            bank[m - 1, center:right] = (right - np.arange(center, right)) / max(right - center, 1)
    return bank


def rms_energy(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    return float(np.sqrt(np.mean(x * x)))


def zero_crossing_rate(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a 1D array with at least 2 samples")
    signs = np.signbit(x)
    return float(np.mean(signs[1:] != signs[:-1]))


def spectral_power(x, frame_length=None, hop_length=None, window="hann", n_fft=None):
    from .core import stft

    spec = stft(x, frame_length=frame_length or 256, hop_length=hop_length, window=window, n_fft=n_fft, pad_end=True)
    return np.abs(spec) ** 2


def _prepare_power_spectrum(x, frame_length=None, hop_length=None, window="hann", n_fft=None):
    power = spectral_power(x, frame_length=frame_length, hop_length=hop_length, window=window, n_fft=n_fft)
    if power.ndim != 2:
        power = np.asarray(power, dtype=float).reshape(1, -1)
    return power


def spectral_flatness(x, frame_length=None, hop_length=None, window="hann", n_fft=None, eps=1e-12):
    power = _prepare_power_spectrum(x, frame_length=frame_length, hop_length=hop_length, window=window, n_fft=n_fft)
    geo_mean = np.exp(np.mean(np.log(power + float(eps)), axis=1))
    arith_mean = np.mean(power, axis=1) + float(eps)
    flatness = geo_mean / arith_mean
    return flatness[0] if flatness.shape[0] == 1 else flatness


def spectral_rolloff(
    x,
    sample_rate=1.0,
    roll_percent=0.85,
    frame_length=None,
    hop_length=None,
    window="hann",
    n_fft=None,
):
    power = _prepare_power_spectrum(x, frame_length=frame_length, hop_length=hop_length, window=window, n_fft=n_fft)
    if not 0.0 < float(roll_percent) < 1.0:
        raise ValueError("roll_percent must be in (0, 1)")
    freqs = np.fft.rfftfreq(power.shape[1] * 2 - 2, d=1.0 / float(sample_rate))
    cumulative = np.cumsum(power, axis=1)
    total = cumulative[:, -1:] + 1e-12
    threshold = float(roll_percent) * total
    idx = np.argmax(cumulative >= threshold, axis=1)
    rolloff = freqs[idx]
    return rolloff[0] if rolloff.shape[0] == 1 else rolloff


def log_mel_spectrogram(
    x,
    sample_rate,
    frame_length=256,
    hop_length=None,
    n_fft=None,
    n_mels=26,
    fmin=0.0,
    fmax=None,
    window="hann",
    eps=1e-12,
):
    mel = mel_spectrogram(
        x,
        sample_rate=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        window=window,
        power=2.0,
    )
    return np.log(np.maximum(mel, float(eps)))


def spectral_centroid(x, sample_rate=1.0, frame_length=None, hop_length=None, window="hann", n_fft=None):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    power = spectral_power(x, frame_length=frame_length, hop_length=hop_length, window=window, n_fft=n_fft)
    freqs = np.fft.rfftfreq(power.shape[1] * 2 - 2, d=1.0 / float(sample_rate))
    denom = np.sum(power, axis=1) + 1e-12
    centroid = np.sum(power * freqs[None, :], axis=1) / denom
    return centroid[0] if centroid.shape[0] == 1 else centroid


def spectral_bandwidth(x, sample_rate=1.0, frame_length=None, hop_length=None, window="hann", n_fft=None):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    power = spectral_power(x, frame_length=frame_length, hop_length=hop_length, window=window, n_fft=n_fft)
    freqs = np.fft.rfftfreq(power.shape[1] * 2 - 2, d=1.0 / float(sample_rate))
    centroid = spectral_centroid(x, sample_rate=sample_rate, frame_length=frame_length, hop_length=hop_length, window=window, n_fft=n_fft)
    centroid = np.asarray(centroid, dtype=float)
    if centroid.ndim == 0:
        centroid = centroid.reshape(1)
    denom = np.sum(power, axis=1) + 1e-12
    spread = np.sum(power * (freqs[None, :] - centroid[:, None]) ** 2, axis=1) / denom
    bandwidth = np.sqrt(np.maximum(spread, 0.0))
    return bandwidth[0] if bandwidth.shape[0] == 1 else bandwidth


def mel_spectrogram(
    x,
    sample_rate,
    frame_length=256,
    hop_length=None,
    n_fft=None,
    n_mels=26,
    fmin=0.0,
    fmax=None,
    window="hann",
    power=2.0,
):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    spec = spectral_power(x, frame_length=frame_length, hop_length=hop_length, window=window, n_fft=n_fft)
    bank = mel_filter_bank(sample_rate=sample_rate, n_fft=spec.shape[1] * 2 - 2, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_spec = spec @ bank.T
    if power != 2.0:
        mel_spec = mel_spec ** (power / 2.0)
    return mel_spec


def _dct_basis(n_in, n_out):
    n_in = int(n_in)
    n_out = int(n_out)
    basis = np.empty((n_out, n_in), dtype=float)
    scale = np.pi / n_in
    for k in range(n_out):
        basis[k] = np.cos(scale * (np.arange(n_in) + 0.5) * k)
    basis[0] *= 1.0 / np.sqrt(n_in)
    if n_out > 1:
        basis[1:] *= np.sqrt(2.0 / n_in)
    return basis


def mfcc(
    x,
    sample_rate,
    n_mfcc=13,
    frame_length=256,
    hop_length=None,
    n_fft=None,
    n_mels=26,
    fmin=0.0,
    fmax=None,
    window="hann",
):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    mel_spec = mel_spectrogram(
        x,
        sample_rate=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        window=window,
        power=2.0,
    )
    log_mel = np.log(np.maximum(mel_spec, 1e-12))
    basis = _dct_basis(log_mel.shape[1], int(n_mfcc))
    coeffs = log_mel @ basis.T
    return coeffs


hz2mel = hz_to_mel
mel2hz = mel_to_hz
mel_bank = mel_filter_bank
spec_pow = spectral_power
spec_cent = spectral_centroid
spec_bw = spectral_bandwidth
mel_spec = mel_spectrogram
log_mel_spec = log_mel_spectrogram
spec_flat = spectral_flatness
spec_rolloff = spectral_rolloff
zcr = zero_crossing_rate
