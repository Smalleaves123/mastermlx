from benchmarks.bench_backend_matrix import (
    BENCHMARK_SCHEMA,
    DEFAULT_MAX_DISTANCE_ERROR,
    DEFAULT_MAX_IIR_ERROR,
    DEFAULT_MAX_TIME_SERIES_ERROR,
    DEFAULT_MAX_CONFUSION_ERROR,
    DEFAULT_MAX_TOP_K_ERROR,
    DEFAULT_MAX_ROC_AUC_ERROR,
    DEFAULT_MAX_AVERAGE_PRECISION_ERROR,
    DEFAULT_MAX_CONTROL_ERROR,
    assert_parity,
    run_backend_matrix,
)
from mastermlx import get_backend
import pytest


def test_backend_matrix_returns_a_versioned_reproducible_parity_record():
    previous_backend = get_backend()

    record = run_backend_matrix(seed=7, repeats=1)

    assert get_backend() == previous_backend
    assert record["schema"] == BENCHMARK_SCHEMA
    assert record["seed"] == 7
    assert record["repeats"] == 1
    assert record["workload"] == {
        "distance_x_shape": [1200, 32],
        "distance_y_shape": [400, 32],
        "iir_samples": 20_000,
        "time_series_samples": 20_000,
        "rolling_window": 128,
        "rolling_variance_ddof": 0,
        "autocorrelation_max_lag": 64,
        "confusion_samples": 100_000,
        "confusion_classes": 8,
        "top_k_samples": 20_000,
        "top_k_classes": 32,
        "top_k": 5,
        "roc_auc_samples": 100_000,
        "average_precision_samples": 100_000,
        "control_states": 8,
        "control_inputs": 3,
        "control_horizon": 128,
    }
    assert [result["backend"] for result in record["results"]][0] == "numpy"
    for key in (
        "distance_seconds",
        "iir_seconds",
        "rolling_mean_seconds",
        "rolling_variance_seconds",
        "autocorrelation_function_seconds",
        "exponential_smoothing_seconds",
        "confusion_matrix_seconds",
        "top_k_accuracy_seconds",
        "roc_auc_seconds",
        "average_precision_seconds",
        "prediction_matrices_seconds",
    ):
        assert all(result[key] >= 0.0 for result in record["results"])
    assert_parity(
        record,
        max_distance_error=DEFAULT_MAX_DISTANCE_ERROR,
        max_iir_error=DEFAULT_MAX_IIR_ERROR,
        max_time_series_error=DEFAULT_MAX_TIME_SERIES_ERROR,
        max_confusion_error=DEFAULT_MAX_CONFUSION_ERROR,
        max_top_k_error=DEFAULT_MAX_TOP_K_ERROR,
        max_roc_auc_error=DEFAULT_MAX_ROC_AUC_ERROR,
        max_average_precision_error=DEFAULT_MAX_AVERAGE_PRECISION_ERROR,
        max_control_error=DEFAULT_MAX_CONTROL_ERROR,
    )


def test_backend_matrix_parity_guard_rejects_numerical_drift():
    record = {
        "results": [
            {
                "backend": "cython",
                "distance_max_error": 2e-10,
                "iir_max_error": 0.0,
                "rolling_mean_max_error": 0.0,
                "rolling_variance_max_error": 0.0,
                "autocorrelation_function_max_error": 0.0,
                "exponential_smoothing_max_error": 0.0,
                "confusion_matrix_max_error": 0.0,
                "top_k_accuracy_max_error": 0.0,
                "roc_auc_max_error": 0.0,
                "average_precision_max_error": 0.0,
                "prediction_matrices_max_error": 0.0,
            }
        ]
    }

    with pytest.raises(RuntimeError, match="distance error"):
        assert_parity(
            record,
            max_distance_error=1e-10,
            max_iir_error=1e-12,
            max_time_series_error=1e-10,
            max_confusion_error=0.0,
            max_top_k_error=0.0,
            max_roc_auc_error=1e-15,
            max_average_precision_error=1e-12,
            max_control_error=1e-12,
        )
