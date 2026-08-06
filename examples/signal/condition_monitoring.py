"""Condition-monitoring workflow for a vibration sensor channel."""

import numpy as np

from mastermlx.signal import (
    OnlineCUSUMDetector,
    SignalFeatureTransformer,
    SignalMonitor,
    StreamingFeatureExtractor,
    VibrationFeatureTransformer,
    signal_quality_report,
    vibration_features,
)


sample_rate = 1000.0
time = np.arange(4000) / sample_rate
healthy = np.sin(2.0 * np.pi * 80.0 * time)
fault = 0.35 * np.sin(2.0 * np.pi * 240.0 * time)
signal = healthy.copy()
signal[2500:] += fault[2500:]

quality = signal_quality_report(
    signal,
    sample_rate=sample_rate,
    saturation_level=2.0,
)
features = vibration_features(signal, sample_rate=sample_rate, n_fft=2048)
print("quality_report:", quality)
print("dominant_frequency:", features["dominant_frequency"])
print("crest_factor:", features["crest_factor"])
print("high_band_energy:", features["band_energy_1"])

transformer = VibrationFeatureTransformer(sample_rate=sample_rate, n_fft=256)
feature_matrix = transformer.fit_transform(signal.reshape(16, 250))
print("feature_matrix_shape:", feature_matrix.shape)
print("feature_names:", transformer.feature_names_)

stream_features = StreamingFeatureExtractor(
    lambda frame: np.array([np.sqrt(np.mean(frame * frame))]),
    frame_length=100,
    hop_length=50,
)
monitor = SignalMonitor(
    stream_features,
    detector=OnlineCUSUMDetector(threshold=0.4, baseline_window=8, cooldown=2),
)
chunks = [signal[:900], signal[900:2100], signal[2100:]]
results = [monitor.push(chunk) for chunk in chunks]
results.append(monitor.flush())
events = [item["events"] for item in results if item["events"].size]
print("monitor_state:", monitor.state())
print("event_positions:", np.concatenate(events) if events else np.empty(0, dtype=int))

# The same transformer can be used by SignalFeatureTransformer in a model pipeline.
model_features = SignalFeatureTransformer(signal_transform=transformer).fit_transform(
    signal.reshape(16, 250)
)
print("model_feature_matrix_shape:", model_features.shape)
