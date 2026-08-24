"""Measure and enforce the cold top-level ``mastermlx`` import budget."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys


_PROBE = r"""
import json
import sys
import time

start = time.perf_counter()
import mastermlx  # noqa: E402,F401
elapsed_ms = (time.perf_counter() - start) * 1000.0

allowed = {
    "mastermlx._lazy_exports",
    "mastermlx.config",
    "mastermlx.version",
}
loaded = {
    name
    for name in sys.modules
    if name.startswith("mastermlx.")
}
print(json.dumps({
    "elapsed_ms": elapsed_ms,
    "loaded": sorted(loaded),
    "numpy_loaded": "numpy" in sys.modules,
    "unexpected": sorted(loaded - allowed),
}))
"""


def _measure_once():
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget-ms",
        type=float,
        default=250.0,
        help="maximum allowed median import time in milliseconds (default: 250)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="number of fresh interpreter measurements (default: 5)",
    )
    args = parser.parse_args()
    if args.budget_ms <= 0.0:
        parser.error("--budget-ms must be positive")
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    results = [_measure_once() for _ in range(args.runs)]
    unexpected = sorted({name for result in results for name in result["unexpected"]})
    if unexpected:
        print("unexpected eager mastermlx modules:", ", ".join(unexpected), file=sys.stderr)
        return 1
    if any(result["numpy_loaded"] for result in results):
        print("top-level import eagerly loaded NumPy", file=sys.stderr)
        return 1

    timings = [float(result["elapsed_ms"]) for result in results]
    median_ms = statistics.median(timings)
    print(
        "mastermlx import "
        f"median={median_ms:.3f} ms "
        f"min={min(timings):.3f} ms "
        f"max={max(timings):.3f} ms "
        f"budget={args.budget_ms:.3f} ms "
        f"modules={len(results[-1]['loaded']) + 1}"
    )
    if median_ms > args.budget_ms:
        print("top-level import exceeded its performance budget", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
