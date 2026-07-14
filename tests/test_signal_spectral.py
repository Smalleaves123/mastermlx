import numpy as np
import pytest

from mastermlx.signal import (
    coherence,
    cross_spectrum,
    frequency_response,
    group_delay,
    periodogram,
    welch_psd,
)


def test_welch_psd_recovers_tone_and_power():
    sample_rate = 1000.0
    t = np.arange(4000) / sample_rate
    x = np.sin(2.0 * np.pi * 125.0 * t)
    freqs, power = welch_psd(x, sample_rate=sample_rate, nperseg=500, nfft=1000)

    peak = int(np.argmax(power[1:]) + 1)
    assert np.isclose(freqs[peak], 125.0)
    assert np.isclose(np.trapezoid(power, freqs), 0.5, atol=0.05)


def test_cross_spectrum_and_coherence_for_scaled_signals():
    sample_rate = 800.0
    t = np.arange(3200) / sample_rate
    x = np.sin(2.0 * np.pi * 80.0 * t)
    y = 2.0 * x
    freqs, cross = cross_spectrum(x, y, sample_rate=sample_rate, nperseg=400, nfft=800)
    _, coh = coherence(x, y, sample_rate=sample_rate, nperseg=400, nfft=800)
    idx = int(np.argmin(np.abs(freqs - 80.0)))

    assert cross[idx].real > 0.0
    assert coh[idx] > 0.99
    assert np.all((coh >= 0.0) & (coh <= 1.0))


def test_periodogram_and_frequency_response():
    x = np.cos(2.0 * np.pi * np.arange(64) / 8.0)
    freqs, power = periodogram(x, sample_rate=64.0)
    assert freqs.shape == power.shape

    freq, response = frequency_response([1.0, -1.0], n_freqs=257, sample_rate=1000.0)
    assert np.isclose(abs(response[0]), 0.0)
    assert np.isclose(abs(response[-1]), 2.0, atol=1e-10)
    assert freq[-1] == 500.0


def test_group_delay_of_one_sample_delay():
    _, delay = group_delay([0.0, 1.0], n_freqs=129, sample_rate=1000.0)
    assert np.allclose(delay[5:-5], 1.0, atol=1e-8)


def test_spectral_validation():
    with pytest.raises(ValueError, match="nfft"):
        welch_psd(np.ones(10), nperseg=8, nfft=4)
    with pytest.raises(ValueError, match="same length"):
        cross_spectrum(np.ones(10), np.ones(9))
    with pytest.raises(ValueError, match="non-zero"):
        frequency_response([1.0], [0.0])
