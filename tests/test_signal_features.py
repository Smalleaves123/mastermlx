import numpy as np

from mastermlx.signal import (
    bandpass_filter,
    de_emphasis,
    hz_to_mel,
    normalize_signal,
    mel_filter_bank,
    mel_spectrogram,
    mel_to_hz,
    mfcc,
    pre_emphasis,
    rms_energy,
    spectral_bandwidth,
    spectral_centroid,
    spectral_power,
    zero_crossing_rate,
)


def test_signal_basic_features_return_expected_shapes():
    x = np.array([1.0, -1.0, 1.0, -1.0])

    assert np.isclose(rms_energy(x), 1.0)
    assert np.isclose(zero_crossing_rate(x), 1.0)


def test_signal_spectral_features_are_finite():
    x = np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False))

    power = spectral_power(x, frame_length=16, hop_length=8, n_fft=16)
    centroid = spectral_centroid(x, sample_rate=16000, frame_length=16, hop_length=8, n_fft=16)
    bandwidth = spectral_bandwidth(x, sample_rate=16000, frame_length=16, hop_length=8, n_fft=16)

    assert power.ndim == 2
    assert np.all(np.isfinite(power))
    assert np.all(np.isfinite(np.asarray(centroid)))
    assert np.all(np.isfinite(np.asarray(bandwidth)))


def test_signal_pre_emphasis_and_inverse_are_consistent():
    x = np.array([1.0, 0.5, 0.25, 0.125])
    y = pre_emphasis(x, coef=0.9)
    z = de_emphasis(y, coef=0.9)

    assert np.allclose(z, x)


def test_signal_normalization_centers_and_scales():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = normalize_signal(x)

    assert np.isclose(np.mean(y), 0.0)
    assert np.isclose(np.std(y), 1.0)


def test_signal_mel_and_mfcc_tools_return_expected_shapes():
    x = np.sin(np.linspace(0, 4 * np.pi, 128, endpoint=False))
    bank = mel_filter_bank(sample_rate=16000, n_fft=32, n_mels=10)
    mel_spec = mel_spectrogram(x, sample_rate=16000, frame_length=32, hop_length=16, n_fft=32, n_mels=10)
    coeffs = mfcc(x, sample_rate=16000, frame_length=32, hop_length=16, n_fft=32, n_mels=10, n_mfcc=5)

    assert bank.shape == (10, 17)
    assert mel_spec.shape[1] == 10
    assert coeffs.shape[1] == 5
    assert np.all(np.isfinite(coeffs))


def test_signal_bandpass_filter_keeps_band_energy_reasonably():
    sr = 200.0
    t = np.arange(0, 2.0, 1.0 / sr)
    low = np.sin(2.0 * np.pi * 5.0 * t)
    high = 0.5 * np.sin(2.0 * np.pi * 40.0 * t)
    x = low + high

    y = bandpass_filter(x, low_cutoff=4.0, high_cutoff=8.0, sample_rate=sr, num_taps=51)

    assert y.shape == x.shape
    assert rms_energy(y) > 0.2
    assert abs(np.corrcoef(y, low)[0, 1]) > 0.5


def test_mel_frequency_helpers_are_invertible():
    freqs = np.array([0.0, 100.0, 1000.0, 4000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(freqs)), freqs)
