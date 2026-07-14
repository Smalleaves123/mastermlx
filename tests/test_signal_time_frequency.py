import numpy as np
import pytest

from mastermlx.signal import (
    analytic_signal,
    cwt,
    extract_ridge,
    hilbert_transform,
    instantaneous_amplitude,
    instantaneous_features,
    instantaneous_frequency,
    instantaneous_phase,
    wavelet_power,
    wavelet_scales,
)


def test_hilbert_and_instantaneous_features_for_a_tone():
    sample_rate = 1024.0
    frequency = 64.0
    t = np.arange(1024) / sample_rate
    x = np.cos(2.0 * np.pi * frequency * t)

    analytic = analytic_signal(x)
    features = instantaneous_features(x, sample_rate=sample_rate)

    assert np.allclose(analytic, np.exp(1j * 2.0 * np.pi * frequency * t), atol=1e-10)
    assert np.allclose(hilbert_transform(x), np.sin(2.0 * np.pi * frequency * t), atol=1e-10)
    assert np.allclose(instantaneous_amplitude(x), 1.0, atol=1e-10)
    assert np.allclose(instantaneous_frequency(x, sample_rate=sample_rate), frequency, atol=1e-10)
    assert np.allclose(features["amplitude"], 1.0, atol=1e-10)
    assert np.allclose(features["frequency"], frequency, atol=1e-10)
    assert np.allclose(
        instantaneous_phase(x, unwrap=True),
        features["phase"],
        atol=1e-12,
    )


def test_cwt_tracks_morlet_frequency_and_returns_power():
    sample_rate = 1024.0
    frequency = 64.0
    t = np.arange(2048) / sample_rate
    x = np.cos(2.0 * np.pi * frequency * t)
    target_scale = 6.0 * sample_rate / (2.0 * np.pi * frequency)
    scales = np.array([10.0, target_scale, 20.0, 30.0])

    returned_scales, frequencies, coefficients = cwt(x, scales, sample_rate=sample_rate)
    power = wavelet_power(coefficients)
    mean_power = np.mean(power[:, 200:-200], axis=1)

    assert np.allclose(returned_scales, scales)
    assert coefficients.shape == (scales.size, x.size)
    assert power.shape == coefficients.shape
    assert np.isclose(frequencies[np.argmax(mean_power)], frequency, atol=1.0)
    assert np.all(power >= 0.0)


def test_wavelet_scales_and_ridge_extraction():
    scales = wavelet_scales(n_scales=8, min_scale=2.0, max_scale=32.0)
    assert scales.shape == (8,)
    assert np.all(np.diff(scales) > 0.0)

    frequencies = np.linspace(10.0, 160.0, 151)
    times = np.arange(80, dtype=float) / 10.0
    target = 30.0 + 0.8 * np.arange(times.size)
    power = np.exp(-0.5 * ((frequencies[:, None] - target[None, :]) / 2.0) ** 2)
    ridge = extract_ridge(power, frequencies, smoothness=0.2, max_jump=3, times=times)

    assert ridge["indices"].shape == times.shape
    assert np.allclose(ridge["time"], times)
    assert np.max(np.abs(ridge["frequencies"] - target)) <= 2.0
    assert np.allclose(ridge["power"], power[ridge["indices"], np.arange(times.size)])


def test_time_frequency_validation():
    with pytest.raises(ValueError, match="real-valued"):
        analytic_signal(np.ones(8, dtype=complex))
    with pytest.raises(ValueError, match="at least the signal length"):
        analytic_signal(np.ones(8), n_fft=4)
    with pytest.raises(ValueError, match="positive"):
        wavelet_scales(min_scale=0.0)
    with pytest.raises(ValueError, match="monotonic"):
        extract_ridge(np.ones((3, 4)), [1.0, 3.0, 2.0])
