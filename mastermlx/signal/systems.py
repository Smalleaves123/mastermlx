"""Discrete-time linear systems, IIR filters, and filter verification."""

from __future__ import annotations

import numpy as np

from ..accel.signal_ops import iir_filter_1d


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


def _response(b, a, n_freqs, sample_rate, whole, check_poles=True):
    freq, omega = _grid(n_freqs, sample_rate, whole)
    z_b = np.exp(-1j * omega[:, None] * np.arange(b.size)[None, :])
    z_a = np.exp(-1j * omega[:, None] * np.arange(a.size)[None, :])
    numerator = z_b @ b
    denominator = z_a @ a
    if check_poles and np.any(np.abs(denominator) < 1e-14):
        raise ValueError("frequency response contains a pole on the evaluation grid")
    with np.errstate(divide="ignore", invalid="ignore"):
        response = np.divide(numerator, denominator)
    return freq, response


def frequency_response(b, a=(1.0,), n_freqs=512, sample_rate=1.0, whole=False):
    """Evaluate ``H(z)=B(z)/A(z)`` on a uniform frequency grid.

    Coefficients are ordered by increasing delay, so the transfer function is
    ``(b[0] + b[1] z^-1 + ...) / (a[0] + a[1] z^-1 + ...)``.
    """

    b = _coeffs(b, "b")
    a = _coeffs(a, "a")
    if a[0] == 0:
        raise ValueError("a[0] must be non-zero")
    return _response(b, a, n_freqs, sample_rate, whole)


def magnitude_response(b, a=(1.0,), n_freqs=512, sample_rate=1.0, whole=False, db=False):
    """Return the magnitude response, optionally in decibels."""

    freq, response = frequency_response(b, a, n_freqs, sample_rate, whole)
    magnitude = np.abs(response)
    if db:
        magnitude = 20.0 * np.log10(np.maximum(magnitude, 1e-12))
    return freq, magnitude


def phase_response(b, a=(1.0,), n_freqs=512, sample_rate=1.0, whole=False, unwrap=True):
    """Return the phase response in radians."""

    freq, response = frequency_response(b, a, n_freqs, sample_rate, whole)
    phase = np.angle(response)
    return freq, np.unwrap(phase) if unwrap else phase


def group_delay(b, a=(1.0,), n_freqs=512, sample_rate=1.0, whole=False):
    """Return the numerical group delay ``-d phase / d omega``."""

    freq, phase = phase_response(b, a, n_freqs, sample_rate, whole, unwrap=True)
    omega = 2.0 * np.pi * freq / float(sample_rate)
    return freq, -np.gradient(phase, omega)


def pole_zero(b, a=(1.0,)):
    """Return zeros, poles, and the leading transfer-function gain."""

    b = _coeffs(b, "b")
    a = _coeffs(a, "a")
    if a[0] == 0:
        raise ValueError("a[0] must be non-zero")
    if b.size <= 1 or not np.any(np.abs(b) > 0.0):
        zeros = np.empty(0, dtype=complex)
    else:
        zeros = np.roots(np.trim_zeros(b, trim="f"))
    poles = np.roots(a) if a.size > 1 else np.empty(0, dtype=complex)
    return zeros, poles, b[0] / a[0]


def is_stable(a, margin=0.0):
    """Return whether all causal IIR poles are strictly inside the unit circle."""

    a = _coeffs(a, "a")
    margin = float(margin)
    if not 0.0 <= margin < 1.0:
        raise ValueError("margin must be in [0, 1)")
    _, poles, _ = pole_zero([1.0], a)
    return bool(np.all(np.abs(poles) < 1.0 - margin))


def check_stability(a, margin=0.0):
    """Validate IIR stability and return its poles."""

    a = _coeffs(a, "a")
    _, poles, _ = pole_zero([1.0], a)
    if not is_stable(a, margin=margin):
        radius = float(np.max(np.abs(poles))) if poles.size else 0.0
        raise ValueError(f"IIR filter is unstable: maximum pole radius is {radius:.6g}")
    return poles


def iir_filter(x, b, a=(1.0,)):
    """Apply a causal IIR filter using the standard difference equation."""

    x = np.asarray(x)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    b = _coeffs(b, "b")
    a = _coeffs(a, "a")
    if a[0] == 0:
        raise ValueError("a[0] must be non-zero")
    dtype = np.result_type(x.dtype, b.dtype, a.dtype, np.float64)
    x = x.astype(dtype, copy=False)
    b = b.astype(dtype, copy=False) / a[0]
    a = a.astype(dtype, copy=False) / a[0]
    if not np.iscomplexobj(x) and not np.iscomplexobj(b) and not np.iscomplexobj(a):
        return iir_filter_1d(
            np.ascontiguousarray(x, dtype=float),
            np.ascontiguousarray(b, dtype=float),
            np.ascontiguousarray(a, dtype=float),
        )
    y = np.zeros(x.size, dtype=dtype)
    for n in range(x.size):
        value = 0.0
        for k in range(min(b.size, n + 1)):
            value += b[k] * x[n - k]
        for k in range(1, min(a.size, n + 1)):
            value -= a[k] * y[n - k]
        y[n] = value
    return np.real_if_close(y)


def zero_phase_filter(x, b, a=(1.0,)):
    """Apply an IIR/FIR filter forward and backward for zero phase."""

    x = np.asarray(x)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")
    return iir_filter(iir_filter(x, b, a)[::-1], b, a)[::-1]


def _butterworth_poles(order):
    angles = np.pi * (2.0 * np.arange(order) + 1.0 + order) / (2.0 * order)
    return np.exp(1j * angles)


def _real_coeffs(values):
    values = np.asarray(values)
    scale = max(1.0, float(np.max(np.abs(values))))
    if np.max(np.abs(np.imag(values))) > 1e-10 * scale:
        raise ValueError("filter design produced non-real coefficients")
    return np.real(values).astype(float)


def _digital_filter(poles, zeros, sample_rate, z_ref):
    scale = 2.0 * float(sample_rate)
    poles = np.asarray(poles, dtype=complex)
    digital_poles = (scale + poles) / (scale - poles)
    b = _real_coeffs(np.poly(np.asarray(zeros, dtype=complex)))
    a = _real_coeffs(np.poly(digital_poles))
    gain = np.polyval(a, z_ref) / np.polyval(b, z_ref)
    b = _real_coeffs(b * gain)
    return b, a


def _butterworth_section(order, cutoff, sample_rate, highpass=False):
    order = int(order)
    cutoff = float(cutoff)
    sample_rate = float(sample_rate)
    if order < 1:
        raise ValueError("order must be at least 1")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0.0 < cutoff < sample_rate / 2.0:
        raise ValueError("cutoff must be between 0 and sample_rate / 2")

    warped = 2.0 * sample_rate * np.tan(np.pi * cutoff / sample_rate)
    prototype = _butterworth_poles(order)
    poles = warped / prototype if highpass else warped * prototype
    zeros = np.ones(order) if highpass else -np.ones(order)
    return _digital_filter(poles, zeros, sample_rate, -1.0 if highpass else 1.0)


def _butterworth_band(order, cutoff, sample_rate, bandstop=False):
    order = int(order)
    sample_rate = float(sample_rate)
    cutoff = np.asarray(cutoff, dtype=float).ravel()
    if order < 1:
        raise ValueError("order must be at least 1")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if (
        cutoff.size != 2
        or not 0.0 < cutoff[0] < cutoff[1] < sample_rate / 2.0
    ):
        raise ValueError("band filters require two cutoffs between 0 and Nyquist")

    warped = 2.0 * sample_rate * np.tan(np.pi * cutoff / sample_rate)
    bandwidth = warped[1] - warped[0]
    center = np.sqrt(warped[0] * warped[1])
    prototype = _butterworth_poles(order)

    if bandstop:
        poles = np.concatenate(
            [np.roots([pole, -bandwidth, pole * center**2]) for pole in prototype]
        )
        center_zero = (2.0 * sample_rate + 1j * center) / (2.0 * sample_rate - 1j * center)
        zeros = np.concatenate(
            [np.full(order, center_zero), np.full(order, np.conj(center_zero))]
        )
        z_ref = 1.0
    else:
        poles = np.concatenate(
            [np.roots([1.0, -bandwidth * pole, center**2]) for pole in prototype]
        )
        zeros = np.concatenate([np.ones(order), -np.ones(order)])
        z_ref = (2.0 * sample_rate + 1j * center) / (2.0 * sample_rate - 1j * center)
    return _digital_filter(poles, zeros, sample_rate, z_ref)


def butterworth(order, cutoff, sample_rate, btype="lowpass"):
    """Design a digital Butterworth IIR filter with bilinear transformation.

    ``lowpass`` and ``highpass`` use an order-``order`` prototype.  Bandpass
    and bandstop designs use the standard analog frequency transformations,
    so their final digital order is ``2 * order``.
    """

    btype = str(btype).lower()
    if btype in {"low", "lowpass"}:
        return _butterworth_section(order, cutoff, sample_rate, highpass=False)
    if btype in {"high", "highpass"}:
        return _butterworth_section(order, cutoff, sample_rate, highpass=True)
    if btype not in {"bandpass", "bandstop", "band"}:
        raise ValueError("btype must be lowpass, highpass, bandpass, or bandstop")
    return _butterworth_band(order, cutoff, sample_rate, bandstop=btype == "bandstop")


design_iir = butterworth
butterworth_filter = butterworth
filtfilt = zero_phase_filter
freq_response = frequency_response


def _band_mask(freq, band, name, sample_rate):
    try:
        lo, hi = (float(value) for value in band)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain two frequencies") from None
    if not 0.0 <= lo < hi <= sample_rate / 2.0:
        raise ValueError(f"{name} must be increasing and within [0, Nyquist]")
    mask = (freq >= lo) & (freq <= hi)
    if not np.any(mask):
        raise ValueError(f"{name} does not intersect the frequency grid")
    return mask


def verify_filter(b, a=(1.0,), sample_rate=1.0, passband=None, stopband=None,
                  n_freqs=2048, tolerance_db=1.0, stopband_db=None):
    """Measure stability, response metrics, and optional band constraints.

    ``tolerance_db`` controls the allowed passband ripple.  ``stopband_db``
    controls the required stopband attenuation and defaults to the same value.
    """

    sample_rate = float(sample_rate)
    tolerance_db = float(tolerance_db)
    stopband_db = tolerance_db if stopband_db is None else float(stopband_db)
    if tolerance_db < 0.0 or stopband_db < 0.0:
        raise ValueError("tolerances must be non-negative")
    b = _coeffs(b, "b")
    a = _coeffs(a, "a")
    if a[0] == 0:
        raise ValueError("a[0] must be non-zero")
    freq, response = _response(b, a, n_freqs, sample_rate, False, check_poles=False)
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))
    phase = np.unwrap(np.angle(response))
    omega = 2.0 * np.pi * freq / sample_rate
    delay = -np.gradient(phase, omega)
    zeros, poles, gain = pole_zero(b, a)
    stable = is_stable(a)
    max_pole_radius = float(np.max(np.abs(poles))) if poles.size else 0.0
    result = {
        "stable": stable,
        "zeros": zeros,
        "poles": poles,
        "gain": gain,
        "max_pole_radius": max_pole_radius,
        "frequency": freq,
        "magnitude_db": magnitude_db,
        "phase": phase,
        "group_delay": delay,
        "pass": stable,
    }
    if passband is not None:
        mask = _band_mask(freq, passband, "passband", sample_rate)
        pass_ripple = float(np.max(magnitude_db[mask]) - np.min(magnitude_db[mask]))
        result["passband_ripple_db"] = pass_ripple
        result["passband_min_db"] = float(np.min(magnitude_db[mask]))
        result["passband_max_db"] = float(np.max(magnitude_db[mask]))
        result["passband_ok"] = pass_ripple <= tolerance_db
        result["pass"] = result["pass"] and result["passband_ok"]
    if stopband is not None:
        mask = _band_mask(freq, stopband, "stopband", sample_rate)
        stop_max = float(np.max(magnitude_db[mask]))
        result["stopband_max_db"] = stop_max
        result["stopband_ok"] = stop_max <= -stopband_db
        result["pass"] = result["pass"] and result["stopband_ok"]
    return result


__all__ = [
    "butterworth",
    "butterworth_filter",
    "check_stability",
    "design_iir",
    "filtfilt",
    "freq_response",
    "frequency_response",
    "group_delay",
    "iir_filter",
    "is_stable",
    "magnitude_response",
    "phase_response",
    "pole_zero",
    "verify_filter",
    "zero_phase_filter",
]
