"""Frequency-domain analysis of discrete-time linear systems."""

from __future__ import annotations

import numpy as np


def _coeffs(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    return values.astype(complex if np.iscomplexobj(values) else float, copy=False)


def _grid(n_freqs, sample_rate, whole):
    n_freqs = int(n_freqs)
    sample_rate = float(sample_rate)
    if n_freqs < 2:
        raise ValueError("n_freqs must be at least 2")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if whole:
        freq = np.arange(n_freqs, dtype=float) * sample_rate / n_freqs
    else:
        freq = np.linspace(0.0, sample_rate / 2.0, n_freqs)
    return freq, 2.0 * np.pi * freq / sample_rate


def frequency_response(b, a=(1.0,), n_freqs=512, sample_rate=1.0, whole=False):
    """Evaluate ``H(z)=B(z)/A(z)`` on a uniformly spaced frequency grid.

    Coefficients are ordered by increasing delay, matching the usual digital
    filter convention: ``b[0] + b[1] z^-1 + ...``.
    """

    b = _coeffs(b, "b")
    a = _coeffs(a, "a")
    if a[0] == 0:
        raise ValueError("a[0] must be non-zero")
    freq, omega = _grid(n_freqs, sample_rate, whole)
    delays_b = np.arange(b.size, dtype=float)
    delays_a = np.arange(a.size, dtype=float)
    z_b = np.exp(-1j * omega[:, None] * delays_b[None, :])
    z_a = np.exp(-1j * omega[:, None] * delays_a[None, :])
    numerator = z_b @ b
    denominator = z_a @ a
    if np.any(np.abs(denominator) < 1e-14):
        raise ValueError("frequency response contains a pole on the evaluation grid")
    return freq, numerator / denominator


def group_delay(b, a=(1.0,), n_freqs=512, sample_rate=1.0, whole=False):
    """Return the numerical group delay ``-d phase / d omega``."""

    freq, response = frequency_response(
        b, a=a, n_freqs=n_freqs, sample_rate=sample_rate, whole=whole
    )
    omega = 2.0 * np.pi * freq / float(sample_rate)
    phase = np.unwrap(np.angle(response))
    return freq, -np.gradient(phase, omega)


freq_response = frequency_response

__all__ = ["freq_response", "frequency_response", "group_delay"]
