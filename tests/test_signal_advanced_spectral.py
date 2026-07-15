import numpy as np
import pytest

from mastermlx.signal import (
    ar_spectrum,
    arma_spectrum,
    cepstrum_peaks,
    complex_cepstrum,
    cross_correlation_peaks,
    cyclic_spectrum,
    envelope_demodulate,
    envelope_spectrum,
    esprit,
    fit_ar,
    fit_arma,
    prony,
    real_cepstrum,
    signal_autocorrelation,
    signal_cross_correlation,
)


def test_real_cepstrum_and_peak_extraction():
    sample_rate = 1000.0
    x = np.zeros(1024)
    x[::16] = 1.0

    quefrency, values = real_cepstrum(x, sample_rate=sample_rate)
    peaks = cepstrum_peaks(
        quefrency,
        values,
        n_peaks=3,
        min_quefrency=0.01,
        max_quefrency=0.08,
    )
    _, complex_values = complex_cepstrum(x, sample_rate=sample_rate)

    assert quefrency.shape == values.shape == complex_values.shape
    assert np.isclose(peaks[0, 0], 0.016)
    assert np.all(np.isfinite(complex_values))


def test_envelope_demodulation_and_envelope_spectrum():
    sample_rate = 1000.0
    t = np.arange(2048) / sample_rate
    x = (1.0 + 0.5 * np.cos(2.0 * np.pi * 12.0 * t)) * np.cos(2.0 * np.pi * 180.0 * t)

    envelope = envelope_demodulate(x, sample_rate=sample_rate, carrier_band=(150.0, 210.0))
    frequencies, spectrum = envelope_spectrum(
        x,
        sample_rate=sample_rate,
        n_fft=4096,
        carrier_band=(150.0, 210.0),
    )
    peak = int(np.argmax(spectrum[1:]) + 1)

    assert np.isclose(frequencies[peak], 12.0, atol=1.0)
    assert np.all(envelope >= 0.0)
    assert np.isfinite(spectrum).all()


def test_cyclic_spectrum_detects_amplitude_modulation():
    sample_rate = 1000.0
    t = np.arange(4096) / sample_rate
    x = (1.0 + 0.5 * np.cos(2.0 * np.pi * 12.0 * t)) * np.cos(2.0 * np.pi * 180.0 * t)

    frequencies, cyclic_frequencies, values = cyclic_spectrum(
        x,
        cyclic_frequencies=[0.0, 24.0],
        sample_rate=sample_rate,
        nperseg=512,
        nfft=1024,
    )
    cyclic_peak = frequencies[np.argmax(np.abs(values[1]))]

    assert values.shape == (2, frequencies.size)
    assert np.allclose(cyclic_frequencies, [0.0, 24.0])
    assert np.isclose(abs(cyclic_peak), 180.0, atol=3.0)


def test_correlation_peak_analysis_finds_delay():
    x = np.zeros(128)
    y = np.zeros(128)
    x[40] = 1.0
    y[47] = 1.0

    lags, values = signal_cross_correlation(x, y, max_lag=20, demean=False)
    peaks = cross_correlation_peaks(x, y, max_lag=20, n_peaks=1, positive_only=True)
    auto_lags, auto_values = signal_autocorrelation(x, max_lag=10, demean=False)

    assert lags[np.argmax(np.abs(values))] == 7
    assert peaks.shape == (1, 2)
    assert peaks[0, 0] == 7
    assert auto_lags[0] == 0
    assert np.isclose(auto_values[0], 1.0)


def test_ar_arma_and_modal_frequency_estimators():
    sample_rate = 1000.0
    t = np.arange(1024) / sample_rate
    tone = np.cos(2.0 * np.pi * 100.0 * t) + 0.2 * np.random.default_rng(0).normal(size=t.size)

    ar, ar_noise = fit_ar(tone, order=8)
    arma, ma, arma_noise = fit_arma(tone, ar_order=4, ma_order=2)
    frequencies, ar_power = ar_spectrum(tone, order=8, sample_rate=sample_rate, n_fft=4096)
    _, arma_power = arma_spectrum(tone, ar_order=4, ma_order=2, sample_rate=sample_rate, n_fft=4096)

    roots = np.exp(1j * 2.0 * np.pi * np.array([80.0, 180.0]) / sample_rate)
    amplitudes = np.array([1.0, 0.5 + 0.2j])
    complex_signal = sum(amplitudes[i] * roots[i] ** np.arange(160) for i in range(2))
    prony_result = prony(complex_signal, order=2, sample_rate=sample_rate)
    esprit_result = esprit(complex_signal, n_components=2, sample_rate=sample_rate)

    assert ar.shape == (8,)
    assert arma.shape == (4,)
    assert ma.shape == (2,)
    assert ar_noise >= 0.0 and arma_noise >= 0.0
    assert np.isclose(frequencies[np.argmax(ar_power)], 100.0, atol=1.0)
    assert np.isclose(frequencies[np.argmax(arma_power)], 100.0, atol=2.0)
    assert np.allclose(prony_result["frequencies"], [80.0, 180.0], atol=1e-6)
    assert np.allclose(esprit_result["frequencies"], [80.0, 180.0], atol=1e-6)


def test_advanced_spectral_validation():
    with pytest.raises(ValueError, match="real-valued"):
        real_cepstrum(np.ones(8, dtype=complex))
    with pytest.raises(ValueError, match="same length"):
        signal_cross_correlation(np.ones(8), np.ones(7))
    with pytest.raises(ValueError, match="carrier_band"):
        envelope_demodulate(np.ones(32), sample_rate=100.0, carrier_band=(10.0, 60.0))
    with pytest.raises(ValueError, match="n_components"):
        esprit(np.ones(8), n_components=4)
