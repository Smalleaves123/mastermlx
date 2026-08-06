# Benchmarks

This directory contains the first executable benchmark entry points for `mastermlx`.

## What each script covers

- `bench_models.py` compares representative estimators against scikit-learn baselines
- `bench_ml_comparison.py` compares estimator quality and runtime against scikit-learn and SciPy numerical primitives
- `bench_accel.py` measures the optional compiled backend against NumPy fallbacks
- `bench_backend_matrix.py` compares NumPy, Cython, and auto/C++ paths with parity errors
- `bench_tabular.py` focuses on the tabular workflow introduced by `TabularExperiment`
- `bench_signal.py` focuses on the signal-processing stack, including pipelines, streaming, detection, and `SignalExperiment`
- `bench_signal_comparison.py` compares Welch PSD, coherence, and Hilbert primitives against SciPy
- `bench_workflows.py` runs a fast smoke benchmark for tabular readiness, signal health, and robot workcell reports
- `bench_graphs.py` compares Python and optional C++ kernels on CSR graph traversal, shortest paths, and analysis
- `bench_robotics.py` compares NumPy and compiled paths for batched robot kinematics, transform composition and pose interpolation, clearance summaries, path-level and detailed collision reports, broad-phase collision checks, trajectory sampling and peaks, retiming, IK, velocity mapping, and bounded planning workers
- `bench_robotics_spatial.py` compares NumPy and C++ paths for arbitrary-axis/RPY URDF kinematics and spatial dynamics
- `bench_robotics_scaling.py` profiles AABB broad-phase scaling, detailed collision-buffer capacity and reuse, plus batched rigid-body dynamics
- `bench_neural.py` compares NumPy and compiled paths for recurrent layers, Conv1D packing, IIR filtering, and ridge extraction

## How to use them

Run the scripts directly from the project root:

```bash
python benchmarks/bench_models.py
PYTHONPATH=. python benchmarks/bench_ml_comparison.py
python benchmarks/bench_accel.py
PYTHONPATH=. python benchmarks/bench_backend_matrix.py --json-output outputs/backend_matrix.json
python benchmarks/bench_tabular.py
python benchmarks/bench_signal.py
PYTHONPATH=. python benchmarks/bench_signal_comparison.py
python benchmarks/bench_workflows.py
python benchmarks/bench_graphs.py
python benchmarks/bench_robotics.py
python benchmarks/bench_robotics_spatial.py
python benchmarks/bench_robotics_scaling.py
PYTHONPATH=. python benchmarks/bench_neural.py
```

`bench_workflows.py` can also export JSON artifacts:

```bash
python benchmarks/bench_workflows.py --output outputs/workflows
```

`bench_ml_comparison.py` can export machine-readable timing and quality results:

```bash
PYTHONPATH=. python benchmarks/bench_ml_comparison.py \
  --json-output outputs/ml_comparison.json
```

For Cython-backed sections, install the development extras first:

```bash
pip install -e ".[dev]"
```

For the SciPy and scikit-learn comparison benchmark, install the comparison
extras:

```bash
pip install -e ".[compare]"
```

`bench_accel.py` switches backends through the public `set_backend()` API and
labels metric sections as fallback paths when Cython is not installed.
For a full NumPy/Cython/auto matrix with parity errors, use
`bench_backend_matrix.py`.

For repeatable SciPy and scikit-learn comparisons, pin BLAS thread counts in
the shell, for example `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

## Benchmarking rules

- Keep runs short enough to be practical in development
- Use fixed seeds for synthetic data
- Report runtime and the task-specific score or summary alongside each section
- Treat these scripts as repeatable smoke benchmarks, not one-off performance claims

## Expected outputs

The scripts print plain-text summaries to stdout. The ML comparison benchmark
also writes JSON when `--json-output` is provided.
