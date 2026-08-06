from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "signal"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, signal)
    axes[0].set_title("Synthetic vibration signal")
    axes[0].set_ylabel("amplitude")
    axes[1].plot(report.windows["start_times"], report.windows["features"][:, 0], marker="o")
    axes[1].set_title("Windowed first feature")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel(report.windows["feature_names"][0])
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    output_path = output_dir / "health_demo.png"
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    print("plot:", output_path)


if __name__ == "__main__":
    main()
