"""Public API example for windowed signal health assessment."""

import numpy as np

from common import check_release
from mastermlx.signal import SignalHealthMonitor, windowed_vibration_features


check_release()
sample_rate = 1000.0
time = np.arange(3000) / sample_rate
signal = np.sin(2.0 * np.pi * 50.0 * time)
signal[2000:] += 0.3 * np.sin(2.0 * np.pi * 220.0 * time[2000:])

monitor = SignalHealthMonitor(
    sample_rate=sample_rate,
    feature_limits={
        "rms": (0.5, 1.2),
        "dominant_frequency": (40.0, 60.0),
        "band_energy_1": (None, 0.02),
    },
    bands=[(0.0, 100.0), (100.0, 500.0)],
    n_fft=512,
)
assessment = monitor.assess(signal)
print("status:", assessment["status"])
print("health_score:", assessment["health_score"])
print("alerts:", assessment["alerts"])
print("violations:", assessment["violations"])

windows = windowed_vibration_features(
    signal,
    sample_rate=sample_rate,
    window_length=500,
    hop_length=250,
    n_fft=512,
    pad_end=True,
)
print("window_start_times:", windows["start_times"])
print("feature_matrix_shape:", windows["features"].shape)
print("feature_names:", windows["feature_names"])

window_health = monitor.assess_batch(
    np.asarray(
        [
            signal[start : start + 500]
            for start in windows["start_samples"]
            if start + 500 <= signal.size
        ]
    )
)
print("window_statuses:", [item["status"] for item in window_health])
