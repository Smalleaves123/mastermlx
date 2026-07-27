import numpy as np

from mastermlx.signal import (
    SignalFeatureTransformer,
    VibrationFeatureTransformer,
    signal_quality_report,
    vibration_features,
)


def test_signal_quality_report_tracks_bad_samples_and_amplitude_metrics():
    signal = np.array([0.0, 1.0, np.nan, np.inf, -1.0])
    report = signal_quality_report(signal, sample_rate=100.0, saturation_level=0.9)

    assert report["n_samples"] == 5
    assert report["finite_samples"] == 3
    assert report["nan_samples"] == 1
    assert report["inf_samples"] == 1
    assert report["valid"] is False
    assert report["duration"] == 0.05
    assert report["saturated_samples"] == 2


def test_vibration_features_recovers_tone_and_band_energy():
    sample_rate = 1000.0
    time = np.arange(1000) / sample_rate
    signal = np.sin(2.0 * np.pi * 125.0 * time)
    features = vibration_features(signal, sample_rate=sample_rate, n_fft=1000)

    assert np.isclose(features["dominant_frequency"], 125.0)
    assert features["rms"] > 0.6
    assert features["crest_factor"] > 1.0
    assert features["band_energy_0"] > 0.9
    assert np.isfinite(features["spectral_centroid"])


def test_vibration_transformer_returns_stable_feature_matrix():
    sample_rate = 800.0
    time = np.arange(256) / sample_rate
    signals = np.asarray([
        np.sin(2.0 * np.pi * 80.0 * time),
        np.sin(2.0 * np.pi * 120.0 * time),
    ])
    transformer = VibrationFeatureTransformer(sample_rate=sample_rate, n_fft=256)
    matrix = transformer.fit_transform(signals)

    assert matrix.shape == (2, len(transformer.feature_names_))
    assert np.all(np.isfinite(matrix))
    assert transformer.transform(signals[0]).shape == (matrix.shape[1],)

    feature_matrix = SignalFeatureTransformer(signal_transform=transformer).fit_transform(signals)
    assert feature_matrix.shape == matrix.shape
