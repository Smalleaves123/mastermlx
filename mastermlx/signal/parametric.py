"""Parametric spectral estimation and modal frequency analysis."""

from __future__ import annotations

import numpy as np


def _signal(x):
    arr = np.asarray(x)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    return arr.astype(complex if np.iscomplexobj(arr) else float, copy=False)


def _center(x, demean):
    return x - np.mean(x) if demean else x


def _real_if_close(values):
    return np.real_if_close(values, tol=1000)


def fit_ar(x, order=8, demean=True):
    """Fit an AR model using the Yule-Walker equations.

    The model convention is ``x[n] = sum(ar[k] * x[n-k-1]) + e[n]``.
    Returns ``(ar_coefficients, innovation_variance)``.
    """

    x = _signal(x)
    order = int(order)
    if order < 1 or order >= x.size:
        raise ValueError("order must satisfy 1 <= order < len(x)")
    x = _center(x, demean)
    autocov = np.empty(order + 1, dtype=complex)
    for lag in range(order + 1):
        autocov[lag] = np.mean(x[lag:] * np.conj(x[: x.size - lag]))
    matrix = np.empty((order, order), dtype=complex)
    for row in range(order):
        for col in range(order):
            lag = row - col
            matrix[row, col] = autocov[lag] if lag >= 0 else np.conj(autocov[-lag])
    try:
        coefficients = np.linalg.solve(matrix, autocov[1:])
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(matrix) @ autocov[1:]
    residual = np.asarray(x[order:], dtype=complex).copy()
    for lag, coefficient in enumerate(coefficients, start=1):
        residual -= coefficient * x[order - lag : x.size - lag]
    variance = float(np.mean(np.abs(residual) ** 2))
    return _real_if_close(coefficients), variance


def _ar_model_spectrum(ar, ma, noise_variance, sample_rate, n_fft):
    ar = np.asarray(ar)
    ma = np.asarray(ma)
    sample_rate = float(sample_rate)
    n_fft = int(n_fft)
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    if n_fft < 2:
        raise ValueError("n_fft must be at least 2")
    if noise_variance < 0.0:
        raise ValueError("noise_variance must be non-negative")
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    z = np.exp(-2j * np.pi * frequencies / sample_rate)
    denominator = np.ones_like(z, dtype=complex)
    numerator = np.ones_like(z, dtype=complex)
    for lag, coefficient in enumerate(ar, start=1):
        denominator -= coefficient * z**lag
    for lag, coefficient in enumerate(ma, start=1):
        numerator += coefficient * z**lag
    power = (float(noise_variance) / sample_rate) * np.abs(numerator / denominator) ** 2
    return frequencies, power


def ar_spectrum(x, order=8, sample_rate=1.0, n_fft=512, demean=True):
    """Estimate a one-sided AR power spectrum."""

    coefficients, variance = fit_ar(x, order=order, demean=demean)
    return _ar_model_spectrum(coefficients, [], variance, sample_rate, n_fft)


def fit_arma(x, ar_order=4, ma_order=2, demean=True):
    """Fit an ARMA model with a Hannan-Rissanen style two-stage estimate."""

    x = _signal(x)
    ar_order = int(ar_order)
    ma_order = int(ma_order)
    if ar_order < 0 or ma_order < 0 or ar_order + ma_order < 1:
        raise ValueError("ar_order and ma_order must be non-negative and not both zero")
    if x.size <= max(1, ar_order + ma_order) + 1:
        raise ValueError("x is too short for the requested ARMA orders")
    x = _center(x, demean)
    warmup = max(1, ar_order + ma_order)
    initial_ar, _ = fit_ar(x, order=warmup, demean=False)
    residual = np.zeros(x.size, dtype=complex)
    residual[:warmup] = x[:warmup]
    for index in range(warmup, x.size):
        prediction = sum(
            initial_ar[lag - 1] * x[index - lag] for lag in range(1, warmup + 1)
        )
        residual[index] = x[index] - prediction

    start = max(warmup, ar_order, ma_order)
    rows: list[list[complex]] = []
    target_values: list[complex] = []
    for index in range(start, x.size):
        row = [x[index - lag] for lag in range(1, ar_order + 1)]
        row.extend(residual[index - lag] for lag in range(1, ma_order + 1))
        rows.append(row)
        target_values.append(x[index])
    design = np.asarray(rows, dtype=complex)
    targets = np.asarray(target_values, dtype=complex)
    coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
    ar = coefficients[:ar_order]
    ma = coefficients[ar_order:]
    innovations = targets - design @ coefficients
    variance = float(np.mean(np.abs(innovations) ** 2))
    return _real_if_close(ar), _real_if_close(ma), variance


def arma_spectrum(
    x,
    ar_order=4,
    ma_order=2,
    sample_rate=1.0,
    n_fft=512,
    demean=True,
):
    """Estimate a one-sided ARMA power spectrum."""

    ar, ma, variance = fit_arma(x, ar_order=ar_order, ma_order=ma_order, demean=demean)
    return _ar_model_spectrum(ar, ma, variance, sample_rate, n_fft)


def _component_result(roots, amplitudes, sample_rate):
    roots = np.asarray(roots, dtype=complex)
    amplitudes = np.asarray(amplitudes, dtype=complex)
    order = np.argsort(np.angle(roots))
    roots = roots[order]
    amplitudes = amplitudes[order]
    return {
        "roots": roots,
        "amplitudes": _real_if_close(amplitudes),
        "frequencies": np.angle(roots) * float(sample_rate) / (2.0 * np.pi),
        "damping": np.log(np.maximum(np.abs(roots), 1e-15)) * float(sample_rate),
    }


def prony(x, order, sample_rate=1.0):
    """Estimate damped complex exponentials with Prony's method."""

    x = _signal(x)
    order = int(order)
    if order < 1 or order >= x.size:
        raise ValueError("order must satisfy 1 <= order < len(x)")
    matrix = np.column_stack([x[order - lag - 1 : x.size - lag - 1] for lag in range(order)])
    polynomial = np.linalg.lstsq(matrix, -x[order:], rcond=None)[0]
    roots = np.roots(np.r_[1.0, polynomial])
    time = np.arange(x.size, dtype=float)
    vandermonde = roots[None, :] ** time[:, None]
    amplitudes = np.linalg.lstsq(vandermonde, x, rcond=None)[0]
    result = _component_result(roots, amplitudes, sample_rate)
    result["coefficients"] = _real_if_close(polynomial)
    return result


def esprit(x, n_components, sample_rate=1.0, n_rows=None):
    """Estimate exponential frequencies with the ESPRIT subspace method."""

    x = _signal(x)
    n_components = int(n_components)
    if n_components < 1 or n_components >= x.size // 2:
        raise ValueError("n_components must be between 1 and len(x)//2 - 1")
    n_rows = x.size // 2 if n_rows is None else int(n_rows)
    if n_rows <= n_components or n_rows >= x.size:
        raise ValueError("n_rows must satisfy n_components < n_rows < len(x)")
    snapshots = np.column_stack([x[index : index + n_rows] for index in range(x.size - n_rows + 1)])
    u, _, _ = np.linalg.svd(snapshots, full_matrices=False)
    subspace = u[:, :n_components]
    transition = np.linalg.pinv(subspace[:-1]) @ subspace[1:]
    roots = np.linalg.eigvals(transition)
    time = np.arange(x.size, dtype=float)
    vandermonde = roots[None, :] ** time[:, None]
    amplitudes = np.linalg.lstsq(vandermonde, x, rcond=None)[0]
    return _component_result(roots, amplitudes, sample_rate)


__all__ = [
    "ar_spectrum",
    "arma_spectrum",
    "esprit",
    "fit_ar",
    "fit_arma",
    "prony",
]
