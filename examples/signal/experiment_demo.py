from __future__ import annotations

from mastermlx import LogisticRegression
from mastermlx.signal import MFCCTransformer, NormalizeSignalTransformer, SignalExperiment, SignalPipeline, make_signal_classification_dataset


def main():
    X, y = make_signal_classification_dataset(n_samples=60, sample_rate=8000, duration=0.1, random_state=42)
    signal_transform = SignalPipeline(
        [
            ("normalize", NormalizeSignalTransformer()),
            ("mfcc", MFCCTransformer(sample_rate=8000, frame_length=128, hop_length=64, n_fft=128, n_mfcc=4, n_mels=8)),
        ]
    )

    experiment = SignalExperiment(
        model=LogisticRegression(n_iter=200, lr=0.05, random_state=0),
        signal_transform=signal_transform,
        search=None,
        task="classification",
    )
    experiment.fit(X[:45], y[:45])

    print("train_score:", experiment.score(X[:45], y[:45]))
    print("test_score:", experiment.score(X[45:], y[45:]))
    print("summary:", experiment.summary())


if __name__ == "__main__":
    main()
