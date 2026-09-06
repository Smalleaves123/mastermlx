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
    rolling_variance,
)
from mastermlx.utils import (
    avg_precision_score,
    confusion_matrix,
    roc_auc_score,
    top_k_accuracy_score,
)


BENCHMARK_SCHEMA = "mastermlx.backend-matrix.v6"
DEFAULT_SEED = 42
DEFAULT_REPEATS = 5
DEFAULT_MAX_DISTANCE_ERROR = 1e-10
DEFAULT_MAX_IIR_ERROR = 1e-12
DEFAULT_MAX_TIME_SERIES_ERROR = 1e-10
DEFAULT_MAX_CONFUSION_ERROR = 0.0
DEFAULT_MAX_TOP_K_ERROR = 0.0
DEFAULT_MAX_ROC_AUC_ERROR = 1e-15
DEFAULT_MAX_AVERAGE_PRECISION_ERROR = 1e-12


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
    (
        y_true,
        y_pred,
        labels,
        top_k_true,
        top_k_scores,
        top_k_labels,
        top_k,
        roc_auc_true,
        roc_auc_scores,
    ) = metric_inputs

    def distance():
        return pairwise_squared_euclidean(X, Y)

    def filtering():
        return iir_filter_1d(signal, b, a)

    def rolling():
        return rolling_mean(series, window)

    def rolling_var():
        return rolling_variance(series, window, ddof=0)

    def autocorrelation():
        return autocorrelation_function(series, max_lag)

    def smoothing():
        return exponential_smoothing(series, alpha)

    def confusion():
        return confusion_matrix(y_true, y_pred, labels=labels)

    def top_k_accuracy():
        return top_k_accuracy_score(
            top_k_true,
            top_k_scores,
            k=top_k,
            labels=top_k_labels,
        )

    def roc_auc():
        return roc_auc_score(roc_auc_true, roc_auc_scores)

    def average_precision():
        return avg_precision_score(roc_auc_true, roc_auc_scores)

    distance_time = _measure(distance, repeats=repeats)
    filter_time = _measure(filtering, repeats=repeats)
    rolling_time = _measure(rolling, repeats=repeats)
    rolling_variance_time = _measure(rolling_var, repeats=repeats)
    autocorrelation_time = _measure(autocorrelation, repeats=repeats)
    smoothing_time = _measure(smoothing, repeats=repeats)
    confusion_time = _measure(confusion, repeats=repeats)
    top_k_time = _measure(top_k_accuracy, repeats=repeats)
    roc_auc_time = _measure(roc_auc, repeats=repeats)
    average_precision_time = _measure(average_precision, repeats=repeats)
    distance_value = distance()
    filter_value = filtering()
    rolling_value = rolling()
    rolling_variance_value = rolling_var()
    autocorrelation_value = autocorrelation()
    smoothing_value = smoothing()
    confusion_value = confusion()
    top_k_value = top_k_accuracy()
    roc_auc_value = roc_auc()
    average_precision_value = average_precision()
    result = {
        "backend": name,
        "distance_seconds": distance_time,
        "iir_seconds": filter_time,
        "rolling_mean_seconds": rolling_time,
        "rolling_variance_seconds": rolling_variance_time,
        "autocorrelation_function_seconds": autocorrelation_time,
        "exponential_smoothing_seconds": smoothing_time,
        "confusion_matrix_seconds": confusion_time,
        "top_k_accuracy_seconds": top_k_time,
        "roc_auc_seconds": roc_auc_time,
        "average_precision_seconds": average_precision_time,
        "distance_max_error": _error(distance_value, references["distance"]),
        "iir_max_error": _error(filter_value, references["iir"]),
        "rolling_mean_max_error": _error(rolling_value, references["rolling_mean"]),
        "rolling_variance_max_error": _error(
            rolling_variance_value, references["rolling_variance"]
        ),
        "autocorrelation_function_max_error": _error(
            autocorrelation_value, references["autocorrelation_function"]
        ),
        "exponential_smoothing_max_error": _error(
            smoothing_value, references["exponential_smoothing"]
        ),
        "confusion_matrix_max_error": _error(confusion_value, references["confusion_matrix"]),
        "top_k_accuracy_max_error": _error(top_k_value, references["top_k_accuracy"]),
        "roc_auc_max_error": _error(roc_auc_value, references["roc_auc"]),
        "average_precision_max_error": _error(
            average_precision_value, references["average_precision"]
        ),
    }
    print(
        f"{name:8s} distance={distance_time:8.5f}s iir={filter_time:8.5f}s "
        f"rolling={rolling_time:8.5f}s variance={rolling_variance_time:8.5f}s "
        f"acf={autocorrelation_time:8.5f}s "
        f"smooth={smoothing_time:8.5f}s confusion={confusion_time:8.5f}s "
        f"top-k={top_k_time:8.5f}s roc-auc={roc_auc_time:8.5f}s "
        f"avg-precision={average_precision_time:8.5f}s "
        f"errors=(distance={result['distance_max_error']:.2e}, "
        f"iir={result['iir_max_error']:.2e}, rolling={result['rolling_mean_max_error']:.2e}, "
        f"variance={result['rolling_variance_max_error']:.2e}, "
        f"acf={result['autocorrelation_function_max_error']:.2e}, "
        f"smooth={result['exponential_smoothing_max_error']:.2e}, "
        f"confusion={result['confusion_matrix_max_error']:.2e}, "
        f"top-k={result['top_k_accuracy_max_error']:.2e}, "
        f"roc-auc={result['roc_auc_max_error']:.2e}, "
        f"avg-precision={result['average_precision_max_error']:.2e})"
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
    top_k_samples = 20_000
    top_k_classes = 32
    top_k = 5
    top_k_true = rng.integers(0, top_k_classes, size=top_k_samples)
    top_k_scores = rng.normal(size=(top_k_samples, top_k_classes))
    top_k_labels = np.arange(top_k_classes)
    roc_auc_samples = 100_000
    roc_auc_true = rng.integers(0, 2, size=roc_auc_samples)
    roc_auc_true[:2] = (0, 1)
    roc_auc_scores = rng.normal(size=roc_auc_samples)
    old_backend = get_backend()
    try:
        set_backend("numpy")
        references = {
            "distance": pairwise_squared_euclidean(X, Y),
            "iir": iir_filter_1d(signal, b, a),
            "rolling_mean": rolling_mean(series, window),
            "rolling_variance": rolling_variance(series, window, ddof=0),
            "autocorrelation_function": autocorrelation_function(series, max_lag),
            "exponential_smoothing": exponential_smoothing(series, alpha),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
            "top_k_accuracy": top_k_accuracy_score(
                top_k_true,
                top_k_scores,
                k=top_k,
                labels=top_k_labels,
            ),
            "roc_auc": roc_auc_score(roc_auc_true, roc_auc_scores),
            "average_precision": avg_precision_score(roc_auc_true, roc_auc_scores),
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
                (
                    y_true,
                    y_pred,
                    labels,
                    top_k_true,
                    top_k_scores,
                    top_k_labels,
                    top_k,
                    roc_auc_true,
                    roc_auc_scores,
                ),
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
            "rolling_variance_ddof": 0,
            "autocorrelation_max_lag": max_lag,
            "confusion_samples": int(y_true.size),
            "confusion_classes": int(labels.size),
            "top_k_samples": top_k_samples,
            "top_k_classes": top_k_classes,
            "top_k": top_k,
            "roc_auc_samples": roc_auc_samples,
            "average_precision_samples": roc_auc_samples,
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
    max_top_k_error,
    max_roc_auc_error,
    max_average_precision_error,
):
    """Fail when a backend drifts beyond the recorded numerical contract."""

    max_distance_error = _non_negative_finite(max_distance_error, "max_distance_error")
    max_iir_error = _non_negative_finite(max_iir_error, "max_iir_error")
    max_time_series_error = _non_negative_finite(
        max_time_series_error, "max_time_series_error"
    )
    max_confusion_error = _non_negative_finite(max_confusion_error, "max_confusion_error")
    max_top_k_error = _non_negative_finite(max_top_k_error, "max_top_k_error")
    max_roc_auc_error = _non_negative_finite(max_roc_auc_error, "max_roc_auc_error")
    max_average_precision_error = _non_negative_finite(
        max_average_precision_error, "max_average_precision_error"
    )
    limits = {
        "distance": max_distance_error,
        "iir": max_iir_error,
        "rolling_mean": max_time_series_error,
        "rolling_variance": max_time_series_error,
        "autocorrelation_function": max_time_series_error,
        "exponential_smoothing": max_time_series_error,
        "confusion_matrix": max_confusion_error,
        "top_k_accuracy": max_top_k_error,
        "roc_auc": max_roc_auc_error,
        "average_precision": max_average_precision_error,
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
    parser.add_argument("--max-top-k-error", type=float, default=DEFAULT_MAX_TOP_K_ERROR)
    parser.add_argument("--max-roc-auc-error", type=float, default=DEFAULT_MAX_ROC_AUC_ERROR)
    parser.add_argument(
        "--max-average-precision-error",
        type=float,
        default=DEFAULT_MAX_AVERAGE_PRECISION_ERROR,
    )
    args = parser.parse_args()

    record = run_backend_matrix(seed=args.seed, repeats=args.repeats)
    assert_parity(
        record,
        max_distance_error=args.max_distance_error,
        max_iir_error=args.max_iir_error,
        max_time_series_error=args.max_time_series_error,
        max_confusion_error=args.max_confusion_error,
        max_top_k_error=args.max_top_k_error,
        max_roc_auc_error=args.max_roc_auc_error,
        max_average_precision_error=args.max_average_precision_error,
    )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
