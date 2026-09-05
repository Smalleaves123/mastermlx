from benchmarks.bench_backend_matrix import (
    BENCHMARK_SCHEMA,
    DEFAULT_MAX_DISTANCE_ERROR,
    DEFAULT_MAX_IIR_ERROR,
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
    }
    assert [result["backend"] for result in record["results"]][0] == "numpy"
    assert all(result["distance_seconds"] >= 0.0 for result in record["results"])
    assert all(result["iir_seconds"] >= 0.0 for result in record["results"])
    assert_parity(
        record,
        max_distance_error=DEFAULT_MAX_DISTANCE_ERROR,
        max_iir_error=DEFAULT_MAX_IIR_ERROR,
    )


def test_backend_matrix_parity_guard_rejects_numerical_drift():
    record = {
        "results": [
            {
                "backend": "cython",
                "distance_max_error": 2e-10,
                "iir_max_error": 0.0,
            }
        ]
    }

    with pytest.raises(RuntimeError, match="distance error"):
        assert_parity(record, max_distance_error=1e-10, max_iir_error=1e-12)
