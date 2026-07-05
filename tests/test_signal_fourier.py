import numpy as np

from mastermlx.signal import (
    FourierTransformer,
    InverseFourierTransformer,
    band_energy,
    dominant_frequency,
    fft_spectrum,
    power_spectrum,
    spectral_feature_vector,
    SpectralFeatureTransformer,
    stft_spectrum,
    top_frequency_peaks,
)


def test_fourier_spectrum_and_inverse_roundtrip():
    x = np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False))

    freqs, spectrum = fft_spectrum(x, sample_rate=64.0, n_fft=64, window=None)
    reconstructed = InverseFourierTransformer(n_fft=64, length=x.size, real=True).transform(spectrum)

    assert freqs.shape == spectrum.shape
    assert np.allclose(reconstructed, x, atol=1e-12)


def test_fourier_peaks_and_dominant_frequency():
    sample_rate = 1024.0
    t = np.arange(0, 1.0, 1.0 / sample_rate)
    x = np.sin(2 * np.pi * 64.0 * t) + 0.5 * np.sin(2 * np.pi * 192.0 * t)

    freq, amp, idx = dominant_frequency(x, sample_rate=sample_rate, n_fft=1024, window="hann")
    peaks = top_frequency_peaks(x, sample_rate=sample_rate, n_fft=1024, window="hann", top_k=2)

    assert np.isclose(freq, 64.0, atol=2.0)
    assert amp > 0.0
    assert idx >= 0
    assert peaks.shape == (2, 2)
    assert np.all(np.isfinite(peaks))


def test_fourier_band_energy_and_transformer_outputs():
    sample_rate = 1024.0
    t = np.arange(0, 1.0, 1.0 / sample_rate)
    x = np.sin(2 * np.pi * 96.0 * t) + 0.25 * np.sin(2 * np.pi * 320.0 * t)

    energies = band_energy(x, sample_rate=sample_rate, bands=[(0, 128), (128, 256), (256, 512)], n_fft=1024, window="hann", normalize=True)
    transformer = FourierTransformer(sample_rate=sample_rate, n_fft=1024, window="hann", output="peaks", top_k=3)
    peaks = transformer.transform(x)
    power = power_spectrum(x, sample_rate=sample_rate, n_fft=1024, window="hann")[1]

    assert energies.shape == (3,)
    assert peaks.shape == (3, 2)
    assert power.ndim == 1
    assert np.all(np.isfinite(energies))
    assert np.all(np.isfinite(peaks))


def test_spectral_feature_vector_combines_fft_and_stft():
    sample_rate = 1024.0
    t = np.arange(0, 1.0, 1.0 / sample_rate)
    x = np.sin(2 * np.pi * 64.0 * t) + 0.3 * np.sin(2 * np.pi * 224.0 * t)

    stft_mag = stft_spectrum(x, frame_length=256, hop_length=128, n_fft=256, window="hann", reduce=None)
    features = spectral_feature_vector(
        x,
        sample_rate=sample_rate,
        frame_length=256,
        hop_length=128,
        n_fft=256,
        window="hann",
        fft_output="power",
        stft_reduce=("mean", "std"),
    )
    transformer = SpectralFeatureTransformer(
        sample_rate=sample_rate,
        frame_length=256,
        hop_length=128,
        n_fft=256,
        window="hann",
        fft_output="power",
        stft_reduce=("mean", "std"),
    )
    transformed = transformer.transform(x)

    expected_width = (256 // 2 + 1) * 3
    assert stft_mag.ndim == 2
    assert stft_mag.shape[1] == 256 // 2 + 1
    assert features.shape == (expected_width,)
    assert transformed.shape == features.shape
    assert np.all(np.isfinite(features))
    assert np.allclose(transformed, features)
