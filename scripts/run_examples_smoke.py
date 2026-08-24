"""Run a deterministic, headless smoke subset of the public examples."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    "examples/quickstart/basic_ml.py",
    "examples/regression/regression_pipeline.py",
    "examples/probabilistic/probabilistic_models.py",
    "examples/bandits/bandit_comparison.py",
    "examples/rl/q_learning_demo.py",
    "examples/tabular/readiness_demo.py",
    "examples/classification/compare_models.py",
    "examples/signal/fourier_demo.py",
)
PLOT_OUTPUTS = (
    "examples/outputs/compare_models.png",
    "examples/outputs/signal/fourier_demo.png",
)


def run_example(relative_path, environment):
    print(f"running {relative_path}", flush=True)
    subprocess.run(
        [sys.executable, relative_path],
        check=True,
        cwd=ROOT,
        env=environment,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the smoke-example paths without running them",
    )
    args = parser.parse_args()
    if args.list:
        print("\n".join(EXAMPLES))
        return 0

    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(ROOT)
        if not existing_pythonpath
        else f"{ROOT}{os.pathsep}{existing_pythonpath}"
    )
    for relative_path in EXAMPLES:
        run_example(relative_path, environment)

    missing = [path for path in PLOT_OUTPUTS if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"examples did not create expected plots: {', '.join(missing)}")
    print(f"examples smoke passed ({len(EXAMPLES)} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
