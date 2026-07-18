import numpy as np

from mastermlx import LogisticRegression
from mastermlx.signal import (
    CUSUMDetector,
    MFCCTransformer,
    NormalizeSignalTransformer,
    SignalExperiment,
    SignalMonitor,
    SignalPipeline,
    StreamingFeatureExtractor,
    compare_signal_models,
    make_signal_classification_dataset,
    rms_energy,
)


def test_signal_pipeline_chains_transforms():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    pipe = SignalPipeline(
        [
            ("normalize", NormalizeSignalTransformer()),
            ("mfcc", MFCCTransformer(sample_rate=16000, frame_length=4, hop_length=2, n_fft=4, n_mfcc=2, n_mels=4)),
        ]
    )

    coeffs = pipe.fit_transform(x)

    assert coeffs.shape[1] == 2
    assert np.all(np.isfinite(coeffs))


def test_streaming_feature_extractor_accumulates_chunks():
    x = np.sin(np.linspace(0, 2 * np.pi, 16, endpoint=False))
    extractor = StreamingFeatureExtractor(lambda frame: np.array([rms_energy(frame)]), frame_length=4, hop_length=2)

    first = extractor.push(x[:5])
    second = extractor.push(x[5:11])
    final = extractor.flush()

    total = np.vstack([arr for arr in [first, second, final] if arr.size > 0])

    assert total.shape[1] == 1
    assert total.shape[0] > 0
    assert np.all(np.isfinite(total))


def test_cusum_detector_finds_mean_shift():
    x = np.concatenate([np.zeros(20), np.ones(20) * 3.0])
    detector = CUSUMDetector(threshold=2.0, drift=0.0, baseline_window=10)

    events = detector.transform(x)

    assert events.size > 0
    assert events[0] >= 15


def test_signal_monitor_tracks_global_event_positions():
    class IndexDetector:
        def transform(self, features):
            return np.arange(features.shape[0], dtype=int)

    extractor = StreamingFeatureExtractor(
        lambda frame: np.array([rms_energy(frame)]),
        frame_length=4,
        hop_length=2,
    )
    monitor = SignalMonitor(extractor, IndexDetector())

    first = monitor.push(np.ones(5))
    second = monitor.push(np.ones(4))
    final = monitor.flush()

    features = np.vstack([item["features"] for item in (first, second, final) if item["features"].size])
    events = np.concatenate([item["events"] for item in (first, second, final) if item["events"].size])
    assert features.shape[0] == 3
    assert np.array_equal(events, np.array([0, 1, 2]))
    assert monitor.state() == {"frames_seen": 3, "has_detector": True}

    monitor.reset()
    assert monitor.state()["frames_seen"] == 0


def test_signal_experiment_runs_end_to_end():
    X, y = make_signal_classification_dataset(n_samples=30, sample_rate=8000, duration=0.1, random_state=0)
    signal_transform = SignalPipeline(
        [
            ("normalize", NormalizeSignalTransformer()),
            ("mfcc", MFCCTransformer(sample_rate=8000, frame_length=128, hop_length=64, n_fft=128, n_mfcc=4, n_mels=8)),
        ]
    )

    experiment = SignalExperiment(
        model=LogisticRegression(n_iter=150, lr=0.05, random_state=0),
        signal_transform=signal_transform,
        search=None,
        task="classification",
    )
    experiment.fit(X[:20], y[:20])

    features = experiment.transform_features(X[20:])
    pred = experiment.predict(X[20:])

    assert features.ndim == 2
    assert pred.shape == y[20:].shape
    assert experiment.summary()["task"] == "classification"
    assert experiment.score(X[:20], y[:20]) >= 0.0


def test_compare_signal_models_returns_leaderboard():
    X, y = make_signal_classification_dataset(n_samples=20, sample_rate=8000, duration=0.1, random_state=1)

    result = compare_signal_models(
        [
            ("logreg_a", LogisticRegression(n_iter=100, lr=0.05, random_state=0)),
            ("logreg_b", LogisticRegression(n_iter=100, lr=0.1, random_state=0)),
        ],
        X,
        y,
        signal_transform=SignalPipeline(
            [
                ("normalize", NormalizeSignalTransformer()),
                ("mfcc", MFCCTransformer(sample_rate=8000, frame_length=128, hop_length=64, n_fft=128, n_mfcc=4, n_mels=8)),
            ]
        ),
    )

    assert result["leaderboard"]
    assert result["best_name"] in {"logreg_a", "logreg_b"}
