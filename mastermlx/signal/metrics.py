from __future__ import annotations

import numpy as np


def signal_snr(reference, estimate, eps=1e-12):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    noise = reference - estimate
    signal_power = np.mean(reference * reference)
    noise_power = np.mean(noise * noise)
    return float(10.0 * np.log10((signal_power + eps) / (noise_power + eps)))


def signal_psnr(reference, estimate, peak=None, eps=1e-12):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    if peak is None:
        peak = np.max(np.abs(reference)) if reference.size else 1.0
    mse = np.mean((reference - estimate) ** 2)
    return float(10.0 * np.log10((float(peak) ** 2 + eps) / (mse + eps)))


def frame_energy(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return float(np.mean(x * x))
    if x.ndim == 2:
        return np.mean(x * x, axis=1)
    raise ValueError("x must be a 1D or 2D array")


__all__ = ["frame_energy", "signal_psnr", "signal_snr"]
