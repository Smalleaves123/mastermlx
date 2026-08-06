"""Benchmark signal primitives against SciPy reference implementations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.signal import coherence as scipy_coherence
from scipy.signal import hilbert as scipy_hilbert
from scipy.signal import welch as scipy_welch

from mastermlx.signal import coherence, hilbert, welch_psd
from mastermlx.signal.core import hann_window


def _measure(function, repeats=5):
    function()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        values.append(time.perf_counter() - start)
    return float(np.median(values))


def _relative_error(actual, reference):
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    denominator = max(float(np.linalg.norm(reference)), 1e-15)
    return float(np.linalg.norm(actual - reference) / denominator)


def _compare(name, ours, reference, quality_label):
    ours_time = _measure(ours)
    reference_time = _measure(reference)
    ours_value = ours()
    reference_value = reference()
    error = _relative_error(ours_value, reference_value)
    result = {
        "name": name,
        "mastermlx_seconds": ours_time,
        "scipy_seconds": reference_time,
        "time_ratio": ours_time / reference_time,
        "relative_error": error,
        "quality_label": quality_label,
    }
    print(
        f"{name:16s} mastermlx={ours_time:8.5f}s  scipy={reference_time:8.5f}s  "
        f"time={ours_time / reference_time:6.2f}x  {quality_label}={error:.3e}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, help="write results to a JSON file")
    args = parser.parse_args()

    sample_rate = 8_000.0
    rng = np.random.default_rng(42)
    time_axis = np.arange(16_384) / sample_rate
    signal = (
        np.sin(2.0 * np.pi * 440.0 * time_axis)
        + 0.35 * np.sin(2.0 * np.pi * 1_200.0 * time_axis)
        + 0.1 * rng.normal(size=time_axis.size)
    )
    second_signal = 0.8 * signal + 0.05 * rng.normal(size=time_axis.size)
    nperseg = 512
    noverlap = 256
    nfft = 1024
    window = hann_window(nperseg)

    print("Signal primitives")
    results = [
        _compare(
            "welch_psd",
            lambda: welch_psd(
                signal,
                sample_rate=sample_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                window=window,
            )[1],
            lambda: scipy_welch(
                signal,
                fs=sample_rate,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                detrend="constant",
                scaling="density",
            )[1],
            "relative_l2_error",
        ),
        _compare(
            "coherence",
            lambda: coherence(
                signal,
                second_signal,
                sample_rate=sample_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                window=window,
            )[1],
            lambda: scipy_coherence(
                signal,
                second_signal,
                fs=sample_rate,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                detrend="constant",
            )[1],
            "relative_l2_error",
        ),
        _compare(
            "hilbert",
            lambda: hilbert(signal),
            lambda: np.imag(scipy_hilbert(signal)),
            "relative_l2_error",
        ),
    ]

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps({"results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
