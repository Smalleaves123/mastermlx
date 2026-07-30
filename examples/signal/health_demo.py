from __future__ import annotations

import numpy as np

from mastermlx.signal import SignalHealthExperiment


def main():
    sample_rate = 1000
    t = np.arange(2048, dtype=float) / sample_rate
    signal = np.sin(2.0 * np.pi * 50.0 * t) + 0.15 * np.sin(2.0 * np.pi * 180.0 * t)

    experiment = SignalHealthExperiment(
        sample_rate=sample_rate,
        feature_limits={
            "rms": (0.1, 1.2),
            "crest_factor": (None, 4.0),
            "dominant_frequency": (40.0, 70.0),
        },
        bands=[(0.0, 100.0), (100.0, 250.0)],
        window_length=256,
        hop_length=128,
    )
    report = experiment.run(signal)

    print("status:", report.summary["status"])
    print("health score:", round(report.summary["health_score"], 2))
    print("alerts:", report.summary["alerts"])
    print("windows:", report.windows["features"].shape[0])


if __name__ == "__main__":
    main()
