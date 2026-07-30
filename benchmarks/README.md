# Benchmarks

This directory contains the first executable benchmark entry points for `mastermlx`.

## What each script covers

- `bench_models.py` compares representative estimators against scikit-learn baselines
- `bench_accel.py` measures the optional compiled backend against NumPy fallbacks
- `bench_tabular.py` focuses on the tabular workflow introduced by `TabularExperiment`
- `bench_signal.py` focuses on the signal-processing stack, including pipelines, streaming, detection, and `SignalExperiment`
- `bench_workflows.py` runs a fast smoke benchmark for tabular readiness, signal health, and robot workcell reports
- `bench_graphs.py` compares Python and optional C++ kernels on CSR graph traversal, shortest paths, and analysis
- `bench_robotics.py` compares NumPy and compiled paths for batched robot kinematics, transform composition and pose interpolation, clearance summaries and detailed collision reports, broad-phase collision checks, trajectory sampling and peaks, retiming, IK, velocity mapping, and bounded planning workers
- `bench_neural.py` compares NumPy and compiled paths for recurrent layers, Conv1D packing, IIR filtering, and ridge extraction

## How to use them

Run the scripts directly from the project root:

```bash
python benchmarks/bench_models.py
python benchmarks/bench_accel.py
python benchmarks/bench_tabular.py
python benchmarks/bench_signal.py
python benchmarks/bench_workflows.py
python benchmarks/bench_graphs.py
python benchmarks/bench_robotics.py
PYTHONPATH=. python benchmarks/bench_neural.py
```

`bench_workflows.py` can also export JSON artifacts:

```bash
python benchmarks/bench_workflows.py --output outputs/workflows
```

For Cython-backed sections, install the development extras first:

```bash
pip install -e ".[dev]"
```

`bench_accel.py` switches backends through the public `set_backend()` API and
labels metric sections as fallback paths when Cython is not installed.

## Benchmarking rules

- Keep runs short enough to be practical in development
- Use fixed seeds for synthetic data
- Report runtime and the task-specific score or summary alongside each section
- Treat these scripts as repeatable smoke benchmarks, not one-off performance claims

## Expected outputs

The scripts print plain-text summaries to stdout.
If you want to keep artifacts, redirect output into `outputs/` or wrap the scripts in a small runner that writes CSV or JSON.
