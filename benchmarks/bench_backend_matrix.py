"""Compare NumPy, Cython, and auto/C++ acceleration paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from mastermlx import get_backend, set_backend
from mastermlx.accel import backend_report, pairwise_squared_euclidean
from mastermlx.accel.signal_ops import iir_filter_1d
from mastermlx.math_tools import (
    autocorrelation_function,
    exponential_smoothing,
    rolling_mean,
)
from mastermlx.utils import confusion_matrix


BENCHMARK_SCHEMA = "mastermlx.backend-matrix.v2"
DEFAULT_SEED = 42
DEFAULT_REPEATS = 5
DEFAULT_MAX_DISTANCE_ERROR = 1e-10
DEFAULT_MAX_IIR_ERROR = 1e-12
DEFAULT_MAX_TIME_SERIES_ERROR = 1e-10
DEFAULT_MAX_CONFUSION_ERROR = 0.0


def _measure(function, repeats=5):
    function()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        values.append(time.perf_counter() - start)
    return float(np.median(values))


def _error(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def _run_backend(
    name,
    distance_inputs,
    signal_inputs,
    time_series_inputs,
    metric_inputs,
    references,
    *,
    repeats,
):
    set_backend(name)
    X, Y = distance_inputs
    signal, b, a = signal_inputs
    series, window, max_lag, alpha = time_series_inputs
    y_true, y_pred, labels = metric_inputs

    def distance():
        return pairwise_squared_euclidean(X, Y)

    def filtering():
        return iir_filter_1d(signal, b, a)

    def rolling():
        return rolling_mean(series, window)

    def autocorrelation():
        return autocorrelation_function(series, max_lag)

    def smoothing():
        return exponential_smoothing(series, alpha)

    def confusion():
        return confusion_matrix(y_true, y_pred, labels=labels)

    distance_time = _measure(distance, repeats=repeats)
    filter_time = _measure(filtering, repeats=repeats)
    rolling_time = _measure(rolling, repeats=repeats)
    autocorrelation_time = _measure(autocorrelation, repeats=repeats)
    smoothing_time = _measure(smoothing, repeats=repeats)
    confusion_time = _measure(confusion, repeats=repeats)
    distance_value = distance()
    filter_value = filtering()
    rolling_value = rolling()
    autocorrelation_value = autocorrelation()
    smoothing_value = smoothing()
    confusion_value = confusion()
    result = {
        "backend": name,
        "distance_seconds": distance_time,
        "iir_seconds": filter_time,
        "rolling_mean_seconds": rolling_time,
        "autocorrelation_function_seconds": autocorrelation_time,
        "exponential_smoothing_seconds": smoothing_time,
        "confusion_matrix_seconds": confusion_time,
        "distance_max_error": _error(distance_value, references["distance"]),
        "iir_max_error": _error(filter_value, references["iir"]),
        "rolling_mean_max_error": _error(rolling_value, references["rolling_mean"]),
        "autocorrelation_function_max_error": _error(
            autocorrelation_value, references["autocorrelation_function"]
        ),
        "exponential_smoothing_max_error": _error(
            smoothing_value, references["exponential_smoothing"]
        ),
        "confusion_matrix_max_error": _error(confusion_value, references["confusion_matrix"]),
    }
    print(
        f"{name:8s} distance={distance_time:8.5f}s iir={filter_time:8.5f}s "
        f"rolling={rolling_time:8.5f}s acf={autocorrelation_time:8.5f}s "
        f"smooth={smoothing_time:8.5f}s confusion={confusion_time:8.5f}s "
        f"errors=(distance={result['distance_max_error']:.2e}, "
        f"iir={result['iir_max_error']:.2e}, rolling={result['rolling_mean_max_error']:.2e}, "
        f"acf={result['autocorrelation_function_max_error']:.2e}, "
        f"smooth={result['exponential_smoothing_max_error']:.2e}, "
        f"confusion={result['confusion_matrix_max_error']:.2e})"
    )
    return result


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise ValueError("repeats must be at least 1")
    return value


def _non_negative_finite(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def run_backend_matrix(*, seed=DEFAULT_SEED, repeats=DEFAULT_REPEATS):
    """Run the fixed backend workload and return a versioned benchmark record."""

    repeats = _positive_int(repeats)
    seed = int(seed)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(1200, 32))
    Y = rng.normal(size=(400, 32))
    signal = rng.normal(size=20_000)
    b = np.array([0.2, 0.1, -0.04])
    a = np.array([1.0, -0.35, 0.08])
    series = rng.normal(size=20_000)
    window = 128
    max_lag = 64
    alpha = 0.2
    y_true = rng.integers(0, 8, size=100_000)
    y_pred = rng.integers(0, 8, size=100_000)
    labels = np.arange(8)
    old_backend = get_backend()
    try:
        set_backend("numpy")
        references = {
            "distance": pairwise_squared_euclidean(X, Y),
            "iir": iir_filter_1d(signal, b, a),
            "rolling_mean": rolling_mean(series, window),
            "autocorrelation_function": autocorrelation_function(series, max_lag),
            "exponential_smoothing": exponential_smoothing(series, alpha),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        }
        report = backend_report()
        backends = ["numpy"]
        if report["available_backends"]["cython"]:
            backends.append("cython")
        backends.append("auto")
        print(json.dumps(report, sort_keys=True))
        results = [
            _run_backend(
                name,
                (X, Y),
                (signal, b, a),
                (series, window, max_lag, alpha),
                (y_true, y_pred, labels),
                references,
                repeats=repeats,
            )
            for name in backends
        ]
    finally:
        set_backend(old_backend)

    return {
        "schema": BENCHMARK_SCHEMA,
        "seed": seed,
        "repeats": repeats,
        "workload": {
            "distance_x_shape": list(X.shape),
            "distance_y_shape": list(Y.shape),
            "iir_samples": int(signal.size),
            "time_series_samples": int(series.size),
            "rolling_window": window,
            "autocorrelation_max_lag": max_lag,
            "confusion_samples": int(y_true.size),
            "confusion_classes": int(labels.size),
        },
        "backend_report": report,
        "results": results,
    }


def assert_parity(
    record,
    *,
    max_distance_error,
    max_iir_error,
    max_time_series_error,
    max_confusion_error,
):
    """Fail when a backend drifts beyond the recorded numerical contract."""

    max_distance_error = _non_negative_finite(max_distance_error, "max_distance_error")
    max_iir_error = _non_negative_finite(max_iir_error, "max_iir_error")
    max_time_series_error = _non_negative_finite(
        max_time_series_error, "max_time_series_error"
    )
    max_confusion_error = _non_negative_finite(max_confusion_error, "max_confusion_error")
    limits = {
        "distance": max_distance_error,
        "iir": max_iir_error,
        "rolling_mean": max_time_series_error,
        "autocorrelation_function": max_time_series_error,
        "exponential_smoothing": max_time_series_error,
        "confusion_matrix": max_confusion_error,
    }
    for result in record["results"]:
        for metric, maximum in limits.items():
            error = result[f"{metric}_max_error"]
            if error > maximum:
                raise RuntimeError(
                    f"{result['backend']} {metric} error {error:.3e} exceeds {maximum:.3e}"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, help="write the matrix to a JSON file")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--max-distance-error", type=float, default=DEFAULT_MAX_DISTANCE_ERROR)
    parser.add_argument("--max-iir-error", type=float, default=DEFAULT_MAX_IIR_ERROR)
    parser.add_argument("--max-time-series-error", type=float, default=DEFAULT_MAX_TIME_SERIES_ERROR)
    parser.add_argument("--max-confusion-error", type=float, default=DEFAULT_MAX_CONFUSION_ERROR)
    args = parser.parse_args()

    record = run_backend_matrix(seed=args.seed, repeats=args.repeats)
    assert_parity(
        record,
        max_distance_error=args.max_distance_error,
        max_iir_error=args.max_iir_error,
        max_time_series_error=args.max_time_series_error,
        max_confusion_error=args.max_confusion_error,
    )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
