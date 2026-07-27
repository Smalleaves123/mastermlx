import numpy as np

from mastermlx.signal import (
    SignalHealthMonitor,
    assess_signal_health,
    windowed_vibration_features,
)


def test_windowed_vibration_features_returns_positions_and_matrix():
    sample_rate = 1000.0
    signal = np.sin(2.0 * np.pi * 50.0 * np.arange(11) / sample_rate)

    result = windowed_vibration_features(
        signal,
        sample_rate=sample_rate,
        window_length=4,
        hop_length=3,
        n_fft=8,
        pad_end=True,
    )

    assert np.array_equal(result["start_samples"], np.array([0, 3, 6, 9]))
    assert np.allclose(result["start_times"], result["start_samples"] / sample_rate)
    assert result["features"].shape == (4, len(result["feature_names"]))
    assert np.all(np.isfinite(result["features"]))


def test_windowed_vibration_features_can_drop_short_tail():
    result = windowed_vibration_features(
        np.ones(3),
        sample_rate=100.0,
        window_length=8,
        n_fft=8,
        pad_end=False,
    )

    assert result["start_samples"].size == 0
    assert result["features"].shape[0] == 0
    assert result["features"].shape[1] == len(result["feature_names"])


def test_assess_signal_health_reports_threshold_violation():
    sample_rate = 1000.0
    signal = np.sin(2.0 * np.pi * 50.0 * np.arange(1000) / sample_rate)

    healthy = assess_signal_health(
        signal,
        sample_rate=sample_rate,
        feature_limits={"rms": (0.5, 1.5), "dominant_frequency": (40.0, 60.0)},
        n_fft=2048,
    )
    warning = assess_signal_health(
        signal,
        sample_rate=sample_rate,
        feature_limits={"rms": (None, 0.5)},
        n_fft=2048,
    )

    assert healthy["status"] == "healthy"
    assert healthy["health_score"] == 100.0
    assert warning["status"] == "warning"
    assert warning["health_score"] == 0.0
    assert "rms" in warning["violations"]


def test_signal_health_monitor_assesses_batches_and_nonfinite_data():
    monitor = SignalHealthMonitor(
        sample_rate=100.0,
        feature_limits={"rms": (0.0, 2.0)},
        n_fft=32,
    )
    batch = monitor.assess_batch(np.ones((2, 32)))
    critical = monitor.assess(np.array([1.0, np.nan]))

    assert len(batch) == 2
    assert all(item["status"] == "healthy" for item in batch)
    assert critical["status"] == "critical"
    assert critical["health_score"] == 0.0
    assert critical["alerts"] == ("non_finite_samples",)
