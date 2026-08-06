# User-Compiled Acceleration

`mastermlx` keeps NumPy implementations as the compatibility baseline and
builds optional C++ and Cython extensions when the local toolchain is ready.

## Check the toolchain

Run this from the repository root:

```bash
conda activate CV
python scripts/build_accel.py --check
```

Use `--json` when the result will be consumed by a script or CI job.

The accelerated build requires NumPy, Cython, pybind11, a C compiler, and a
C++ compiler. Typical host prerequisites are:

- macOS: Xcode Command Line Tools
- Ubuntu/Debian: `build-essential`
- Windows: Visual Studio C++ Build Tools

## Build locally

```bash
python scripts/build_accel.py --build
```

This performs an editable build using the active Python environment and the
development dependencies. It is appropriate for a local Conda environment
such as `CV`.

For a NumPy-only install on a machine without a compiler:

```bash
python scripts/build_accel.py --numpy-only
```

The equivalent environment switch is:

```bash
MASTERML_DISABLE_EXTENSIONS=1 pip install .
```

## Inspect capabilities

```python
from mastermlx.accel import backend_report
from mastermlx.robotics import robotics_backend_report

print(backend_report())
print(robotics_backend_report())
```

`available_backends` reports installed NumPy, Cython, and C++ capabilities
independently of the currently selected backend. `active` reports the path
selected for the current request.

Select a backend explicitly when comparing implementations:

```python
import mastermlx

mastermlx.set_backend("numpy")
mastermlx.set_backend("cython")
mastermlx.set_backend("auto")
```

`auto` selects C++ kernels where available, then Cython, then NumPy fallback.

## Benchmark compiled paths

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/bench_backend_matrix.py \
  --json-output outputs/backend_matrix.json
```

The benchmark reports runtime and maximum absolute parity error for pairwise
distance and IIR signal kernels across NumPy, Cython, and auto/C++ paths.
Only promote a new compiled kernel when it provides a material workload gain
and retains fallback parity.
