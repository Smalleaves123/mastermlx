from __future__ import annotations

import numpy as np

from .augmentation import add_noise, time_shift


def make_sine_wave(frequency=440.0, sample_rate=16000, duration=1.0, amplitude=1.0, phase=0.0):
    sample_rate = float(sample_rate)
    duration = float(duration)
    t = np.arange(int(round(sample_rate * duration)), dtype=float) / sample_rate
    x = float(amplitude) * np.sin(2.0 * np.pi * float(frequency) * t + float(phase))
    return t, x


def make_multi_tone(frequencies, sample_rate=16000, duration=1.0, amplitudes=None):
    sample_rate = float(sample_rate)
    duration = float(duration)
    frequencies = np.asarray(frequencies, dtype=float).ravel()
    if frequencies.size == 0:
        raise ValueError("frequencies must be non-empty")
    if amplitudes is None:
        amplitudes = np.ones(frequencies.size, dtype=float) / frequencies.size
    else:
        amplitudes = np.asarray(amplitudes, dtype=float).ravel()
        if amplitudes.shape != frequencies.shape:
            raise ValueError("amplitudes must match frequencies")
    t = np.arange(int(round(sample_rate * duration)), dtype=float) / sample_rate
    x = np.zeros_like(t)
    for f, a in zip(frequencies, amplitudes):
        x += float(a) * np.sin(2.0 * np.pi * f * t)
    return t, x


def make_chirp(f0=100.0, f1=1000.0, sample_rate=16000, duration=1.0):
    sample_rate = float(sample_rate)
    duration = float(duration)
    t = np.arange(int(round(sample_rate * duration)), dtype=float) / sample_rate
    k = (float(f1) - float(f0)) / max(duration, 1e-12)
    phase = 2.0 * np.pi * (float(f0) * t + 0.5 * k * t * t)
    return t, np.sin(phase)


def make_impulse_train(period, sample_rate=16000, duration=1.0):
    sample_rate = float(sample_rate)
    duration = float(duration)
    n = int(round(sample_rate * duration))
    period = max(1, int(period))
    x = np.zeros(n, dtype=float)
    x[::period] = 1.0
    t = np.arange(n, dtype=float) / sample_rate
    return t, x


def make_signal_classification_dataset(
    n_samples=200,
    sample_rate=16000,
    duration=0.5,
    random_state=None,
    noise_scale=0.08,
):
    """Build a small binary signal dataset for demos and smoke tests."""

    rng = np.random.default_rng(random_state)
    n_samples = int(n_samples)
    sample_rate = float(sample_rate)
    duration = float(duration)
    n_points = max(1, int(round(sample_rate * duration)))

    X = np.empty((n_samples, n_points), dtype=float)
    y = np.empty(n_samples, dtype=int)

    for i in range(n_samples):
        label = int(rng.integers(0, 2))
        t = np.arange(n_points, dtype=float) / sample_rate
        if label == 0:
            freq = rng.uniform(120.0, 260.0)
            signal = np.sin(2.0 * np.pi * freq * t)
        else:
            base = np.sin(2.0 * np.pi * rng.uniform(300.0, 520.0) * t)
            tone2 = 0.35 * np.sin(2.0 * np.pi * rng.uniform(650.0, 900.0) * t)
            sweep = np.sin(2.0 * np.pi * (150.0 * t + 0.5 * rng.uniform(180.0, 320.0) * t * t))
            signal = 0.55 * base + 0.25 * tone2 + 0.2 * sweep
        signal = add_noise(signal, scale=noise_scale, random_state=rng)
        signal = time_shift(signal, shift=int(rng.integers(-n_points // 20, n_points // 20 + 1)))
        X[i] = signal
        y[i] = label

    return X, y


def make_signal_anomaly_series(
    length=4000,
    change_point=2000,
    sample_rate=16000,
    random_state=None,
    base_frequency=180.0,
    shift_frequency=260.0,
    noise_scale=0.05,
):
    """Create a 1D signal with a clear mean/frequency shift for detector demos."""

    rng = np.random.default_rng(random_state)
    length = int(length)
    change_point = int(change_point)
    if length < 2:
        raise ValueError("length must be at least 2")
    if not 0 < change_point < length:
        raise ValueError("change_point must be in (0, length)")
    sample_rate = float(sample_rate)

    t = np.arange(length, dtype=float) / sample_rate
    x = np.sin(2.0 * np.pi * base_frequency * t)
    x[change_point:] = 1.4 * np.sin(2.0 * np.pi * shift_frequency * t[change_point:]) + 0.35
    x = add_noise(x, scale=noise_scale, random_state=rng)
    events = np.zeros(length, dtype=int)
    events[change_point:] = 1
    return x, events, change_point


__all__ = [
    "make_chirp",
    "make_impulse_train",
    "make_multi_tone",
    "make_signal_anomaly_series",
    "make_signal_classification_dataset",
    "make_sine_wave",
]
