"""
Benchmark the signal-processing workflow entry point.

This script focuses on the higher-level pipeline and streaming abstractions that
turn raw arrays into feature vectors or detection events.
"""
from __future__ import annotations

import time

import numpy as np

from mastermlx import LogisticRegression
from mastermlx.signal import (
    FourierTransformer,
    CUSUMDetector,
    MFCCTransformer,
    NormalizeSignalTransformer,
    SpectralFeatureTransformer,
    SignalExperiment,
    SignalPipeline,
    StreamingFeatureExtractor,
    compare_signal_models,
    make_signal_classification_dataset,
    rms_energy,
    zero_crossing_rate,
)


def bench(fn, n_runs=3):
    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), result


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def make_wave(seed=7, n_samples=16_000):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_samples, endpoint=False)
    base = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = 0.15 * rng.normal(size=n_samples)
    return base + noise


def benchmark_pipeline():
    x = make_wave()
    def run_pipeline():
        pipeline = SignalPipeline(
            [
                ("normalize", NormalizeSignalTransformer()),
                (
                    "mfcc",
                    MFCCTransformer(
                        sample_rate=16_000,
                        frame_length=512,
                        hop_length=256,
                        n_fft=512,
                        n_mfcc=8,
                        n_mels=24,
                    ),
                ),
            ]
        )
        return pipeline.fit_transform(x)

    elapsed, coeffs = bench(run_pipeline, n_runs=3)
    print(f"  pipeline fit_transform   {elapsed:8.4f}s  shape={tuple(coeffs.shape)}")
    return coeffs


def benchmark_streaming():
    x = make_wave()

    def features(frame):
        return np.array([rms_energy(frame), zero_crossing_rate(frame)], dtype=float)

    chunk_sizes = (700, 900, 1100, 1300, 1500, 1500, 1500, 1500, 1500, 1500)

    def run_stream():
        extractor = StreamingFeatureExtractor(features, frame_length=512, hop_length=256)
        outputs = []
        cursor = 0
        for size in chunk_sizes:
            chunk = x[cursor : cursor + size]
            cursor += size
            out = extractor.push(chunk)
            if out.size:
                outputs.append(out)
        tail = extractor.flush()
        if tail.size:
            outputs.append(tail)
        return np.vstack(outputs) if outputs else np.empty((0, 2))

    elapsed, matrix = bench(run_stream, n_runs=3)
    print(f"  streaming extraction     {elapsed:8.4f}s  shape={tuple(matrix.shape)}")
    return matrix


def benchmark_detection():
    x = np.concatenate([np.zeros(2000), np.ones(2000) * 2.5, np.ones(2000) * 2.8])
    elapsed, events = bench(lambda: CUSUMDetector(threshold=1.8, drift=0.0, baseline_window=128).transform(x), n_runs=5)
    print(f"  cusum detection          {elapsed:8.4f}s  events={events.size}")
    return events


def benchmark_fourier():
    x = make_wave(n_samples=8192)

    def run_fourier():
        transformer = FourierTransformer(sample_rate=8192, n_fft=8192, window="hann", output="peaks", top_k=5)
        return transformer.transform(x)

    elapsed, peaks = bench(run_fourier, n_runs=3)
    print(f"  fourier peaks            {elapsed:8.4f}s  shape={tuple(peaks.shape)}")
    return peaks


def benchmark_spectral_features():
    x = make_wave(n_samples=8192)

    def run_spectral():
        transformer = SpectralFeatureTransformer(
            sample_rate=8192,
            frame_length=512,
            hop_length=256,
            n_fft=512,
            window="hann",
            fft_output="power",
            stft_reduce=("mean", "std", "max"),
        )
        return transformer.transform(x)

    elapsed, features = bench(run_spectral, n_runs=3)
    print(f"  spectral features        {elapsed:8.4f}s  width={features.size}")
    return features


def benchmark_signal_experiment():
    X, y = make_signal_classification_dataset(n_samples=180, sample_rate=8000, duration=0.25, random_state=0)
    X_train, X_test = X[:140], X[140:]
    y_train, y_test = y[:140], y[140:]

    signal_transform = SignalPipeline(
        [
            ("normalize", NormalizeSignalTransformer()),
            (
                "mfcc",
                MFCCTransformer(
                    sample_rate=8000,
                    frame_length=256,
                    hop_length=128,
                    n_fft=256,
                    n_mfcc=6,
                    n_mels=20,
                ),
            ),
        ]
    )

    experiment = SignalExperiment(
        model=LogisticRegression(n_iter=400, lr=0.05, random_state=0),
        signal_transform=signal_transform,
        search=None,
        task="classification",
    )
    elapsed, _ = bench(lambda: experiment.fit(X_train, y_train), n_runs=2)
    train_score = experiment.score(X_train, y_train)
    test_score = experiment.score(X_test, y_test)
    print(f"  signal experiment       {elapsed:8.4f}s  train={train_score:6.3f}  test={test_score:6.3f}")

    leaderboard = compare_signal_models(
        [
            ("logreg_a", LogisticRegression(n_iter=400, lr=0.05, random_state=0)),
            ("logreg_b", LogisticRegression(n_iter=400, lr=0.1, random_state=0)),
        ],
        X_train,
        y_train,
        signal_transform=signal_transform,
        task="classification",
    )
    print(f"  leaderboard             {leaderboard['leaderboard']}")
    return experiment


def main():
    section("Signal Pipeline")
    benchmark_pipeline()

    section("Streaming Features")
    benchmark_streaming()

    section("Event Detection")
    benchmark_detection()

    section("Fourier Analysis")
    benchmark_fourier()

    section("Spectral Features")
    benchmark_spectral_features()

    section("Signal Experiment")
    benchmark_signal_experiment()

    section("Summary")
    print("  This benchmark covers the signal workflow from raw samples to Fourier features, events, and a supervised experiment.")
    print("  It is intentionally lightweight so it can stay in the development loop.")


if __name__ == "__main__":
    main()
