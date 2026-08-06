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


def _run_backend(name, distance_inputs, signal_inputs, references):
    set_backend(name)
    X, Y = distance_inputs
    signal, b, a = signal_inputs
    def distance():
        return pairwise_squared_euclidean(X, Y)

    def filtering():
        return iir_filter_1d(signal, b, a)
    distance_time = _measure(distance)
    filter_time = _measure(filtering)
    distance_value = distance()
    filter_value = filtering()
    result = {
        "backend": name,
        "distance_seconds": distance_time,
        "iir_seconds": filter_time,
        "distance_max_error": _error(distance_value, references["distance"]),
        "iir_max_error": _error(filter_value, references["iir"]),
    }
    print(
        f"{name:8s} distance={distance_time:8.5f}s  iir={filter_time:8.5f}s  "
        f"errors=({result['distance_max_error']:.2e}, {result['iir_max_error']:.2e})"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, help="write the matrix to a JSON file")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    X = rng.normal(size=(1200, 32))
    Y = rng.normal(size=(400, 32))
    signal = rng.normal(size=20_000)
    b = np.array([0.2, 0.1, -0.04])
    a = np.array([1.0, -0.35, 0.08])
    old_backend = get_backend()
    try:
        set_backend("numpy")
        references = {
            "distance": pairwise_squared_euclidean(X, Y),
            "iir": iir_filter_1d(signal, b, a),
        }
        report = backend_report()
        backends = ["numpy"]
        if report["available_backends"]["cython"]:
            backends.append("cython")
        backends.append("auto")
        print(json.dumps(report, sort_keys=True))
        results = [
            _run_backend(name, (X, Y), (signal, b, a), references) for name in backends
        ]
    finally:
        set_backend(old_backend)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps({"backend_report": report, "results": results}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
