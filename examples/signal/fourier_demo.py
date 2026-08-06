from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mastermlx.signal import (
    FourierTransformer,
    InverseFourierTransformer,
    band_energy,
    dominant_frequency,
    fft_spectrum,
    make_multi_tone,
    SpectralFeatureTransformer,
    top_frequency_peaks,
)


def main():
    sample_rate = 2048
    duration = 1.0
    _, x = make_multi_tone(
        frequencies=[64.0, 192.0, 320.0],
        sample_rate=sample_rate,
        duration=duration,
        amplitudes=[1.0, 0.6, 0.3],
    )

    freqs, spectrum = fft_spectrum(x, sample_rate=sample_rate, n_fft=2048, window="hann")
    peak_freq, peak_amp, peak_idx = dominant_frequency(x, sample_rate=sample_rate, n_fft=2048, window="hann")
    peaks = top_frequency_peaks(x, sample_rate=sample_rate, n_fft=2048, window="hann", top_k=3)
    bands = band_energy(x, sample_rate=sample_rate, bands=[(0, 128), (128, 256), (256, 512)], n_fft=2048, window="hann", normalize=True)

    transformer = FourierTransformer(sample_rate=sample_rate, n_fft=2048, window="hann", output="peaks", top_k=3)
    peak_features = transformer.transform(x)
    spectral_features = SpectralFeatureTransformer(
        sample_rate=sample_rate,
        frame_length=512,
        hop_length=256,
        n_fft=2048,
        window="hann",
        fft_output="amplitude",
        stft_reduce=("mean", "std", "max"),
    ).transform(x)

    inverse = InverseFourierTransformer(n_fft=2048, length=x.size, real=True)
    reconstructed = inverse.transform(spectrum)
    reconstruction_error = float(np.mean(np.abs(reconstructed - x)))

    print("spectrum_bins:", freqs.shape[0])
    print("dominant_frequency:", peak_freq, "amplitude:", peak_amp, "index:", peak_idx)
    print("top_peaks:\n", peaks)
    print("band_energy:", bands)
    print("peak_features_shape:", peak_features.shape)
    print("spectral_features_shape:", spectral_features.shape)
    print("reconstruction_error:", reconstruction_error)

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "signal"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    axes[0].plot(np.arange(x.size) / sample_rate, x, label="input")
    axes[0].plot(np.arange(reconstructed.size) / sample_rate, reconstructed, "--", label="reconstructed")
    axes[0].set_title("Multi-tone signal and inverse FFT")
    axes[0].set_xlabel("time (s)")
    axes[0].legend()
    axes[1].plot(freqs, spectrum)
    axes[1].set_xlim(0.0, 512.0)
    axes[1].set_title("Amplitude spectrum")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("amplitude")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    output_path = output_dir / "fourier_demo.png"
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    print("plot:", output_path)


if __name__ == "__main__":
    main()
