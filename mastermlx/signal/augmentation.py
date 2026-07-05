from __future__ import annotations

import numpy as np


def add_noise(x, scale=0.01, snr_db=None, random_state=None):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    rng = np.random.default_rng(random_state)
    if snr_db is not None:
        signal_power = np.mean(x * x)
        noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0) + 1e-12)
        scale = np.sqrt(max(noise_power, 0.0))
    return x + rng.normal(scale=float(scale), size=x.shape)


def time_shift(x, shift, fill_value=0.0):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    shift = int(shift)
    out = np.full_like(x, float(fill_value))
    if shift == 0:
        return x.copy()
    if shift > 0:
        out[shift:] = x[:-shift]
    else:
        out[:shift] = x[-shift:]
    return out


def scale_amplitude(x, factor):
    x = np.asarray(x, dtype=float)
    return x * float(factor)


def mixup_signals(x1, x2, alpha=0.2, random_state=None):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("x1 and x2 must have the same shape")
    rng = np.random.default_rng(random_state)
    lam = rng.beta(float(alpha), float(alpha))
    return lam * x1 + (1.0 - lam) * x2


__all__ = ["add_noise", "mixup_signals", "scale_amplitude", "time_shift"]
