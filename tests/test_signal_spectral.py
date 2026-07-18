import numpy as np
import pytest

from mastermlx.signal import (
    butterworth,
    check_stability,
    coherence,
    cross_spectrum,
    frequency_response,
    iir_filter,
    group_delay,
    is_stable,
    phase_response,
    pole_zero,
    periodogram,
    verify_filter,
    welch_psd,
    zero_phase_filter,
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


def test_iir_filter_uses_normalized_difference_equation():
    x = np.array([1.0, 0.0, 0.0, 0.0])
    y = iir_filter(x, [0.5], [2.0, -1.0])
    assert np.allclose(y, [0.25, 0.125, 0.0625, 0.03125])


def test_phase_pole_zero_and_stability_analysis():
    freq, phase = phase_response([1.0, -0.5], [1.0, -0.8], n_freqs=129)
    zeros, poles, gain = pole_zero([1.0, -0.5], [1.0, -0.8])

    assert freq.shape == phase.shape
    assert np.all(np.isfinite(phase))
    assert np.allclose(zeros, [0.5])
    assert np.allclose(poles, [0.8])
    assert np.isclose(gain, 1.0)
    assert is_stable([1.0, -0.8])
    assert not is_stable([1.0, -1.01])
    assert np.allclose(check_stability([1.0, -0.8]), [0.8])
    with pytest.raises(ValueError, match="unstable"):
        check_stability([1.0, -1.01])


def test_zero_phase_fir_has_symmetric_impulse_response():
    x = np.zeros(21)
    x[10] = 1.0
    y = zero_phase_filter(x, [0.25, 0.5, 0.25])

    assert np.allclose(y, y[::-1])
    assert np.isclose(y[10], 0.375)


@pytest.mark.parametrize(
    "btype, cutoff",
    [
        ("lowpass", 100.0),
        ("highpass", 100.0),
        ("bandpass", [100.0, 200.0]),
        ("bandstop", [100.0, 200.0]),
    ],
)
def test_butterworth_designs_are_stable_and_have_expected_order(btype, cutoff):
    b, a = butterworth(4, cutoff, sample_rate=1000.0, btype=btype)
    poles = pole_zero(b, a)[1]

    assert len(b) == len(a)
    expected_size = 5 if btype in {"lowpass", "highpass"} else 9
    assert len(a) == expected_size
    assert np.max(np.abs(poles)) < 1.0


def test_butterworth_frequency_shapes_and_verification_report():
    low_b, low_a = butterworth(4, 100.0, sample_rate=1000.0, btype="lowpass")
    high_b, high_a = butterworth(4, 100.0, sample_rate=1000.0, btype="highpass")
    band_b, band_a = butterworth(4, [100.0, 200.0], sample_rate=1000.0, btype="bandpass")
    stop_b, stop_a = butterworth(4, [100.0, 200.0], sample_rate=1000.0, btype="bandstop")

    freq, low = frequency_response(low_b, low_a, n_freqs=1001, sample_rate=1000.0)
    _, high = frequency_response(high_b, high_a, n_freqs=1001, sample_rate=1000.0)
    _, band = frequency_response(band_b, band_a, n_freqs=1001, sample_rate=1000.0)
    _, stop = frequency_response(stop_b, stop_a, n_freqs=1001, sample_rate=1000.0)

    def at(values, hz):
        return float(abs(values[np.argmin(abs(freq - hz))]))
    assert np.isclose(at(low, 0.0), 1.0, atol=1e-8)
    assert at(low, 300.0) < 0.01
    assert np.isclose(at(high, 500.0), 1.0, atol=1e-8)
    assert at(high, 20.0) < 0.01
    assert at(band, 150.0) > at(band, 50.0)
    assert at(band, 150.0) > at(band, 300.0)
    assert at(stop, 150.0) < 0.01
    assert at(stop, 50.0) > 0.9

    report = verify_filter(
        low_b,
        low_a,
        sample_rate=1000.0,
        passband=(0.0, 50.0),
        stopband=(200.0, 500.0),
        tolerance_db=1.0,
        stopband_db=20.0,
    )
    assert report["pass"]
    assert report["stable"]
    assert report["stopband_ok"]
    assert report["phase"].shape == report["group_delay"].shape == report["frequency"].shape


def test_spectral_validation():
    with pytest.raises(ValueError, match="nfft"):
        welch_psd(np.ones(10), nperseg=8, nfft=4)
    with pytest.raises(ValueError, match="same length"):
        cross_spectrum(np.ones(10), np.ones(9))
    with pytest.raises(ValueError, match="non-zero"):
        frequency_response([1.0], [0.0])
    with pytest.raises(ValueError, match="Nyquist"):
        butterworth(2, [100.0, 600.0], sample_rate=1000.0, btype="bandpass")
