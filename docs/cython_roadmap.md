# Cython / C++ Optimization Roadmap

This roadmap describes how `mastermlx` should continue adopting compiled
hot-path kernels in a way that stays close to the `numpy` / `scikit-learn`
development model.

## Working principles

- Keep Python as the public API layer.
- Move repeated numeric loops into Cython or C++ only when profiling shows a clear win.
- Preserve NumPy fallbacks so `pip install mastermlx` still works without a compiler.
- Keep validation in Python when it improves error messages or reduces kernel complexity.
- Benchmark every compiled path against its NumPy fallback.

## Current compiled coverage

- `mastermlx.accel`
  - pairwise distances
  - kernels
  - tree split search
  - convolution and max pooling
- `mastermlx.control`
  - iLQR rollout
  - finite-difference Jacobians
  - trajectory cost accumulation
- `mastermlx.robotics`
  - packed forward kinematics
  - packed geometric Jacobians
  - inverse-kinematics helper packing
  - trajectory sampling and joint-path smoothing

## Next priority batches

### Batch 1: math tools

Target modules:

- `mastermlx.math_tools.time_series`
- `mastermlx.utils.metrics`
- `mastermlx.utils.distance`
- `mastermlx.utils.kernels`

Why:

- these functions are called from many estimators and ML models
- they are mostly array loops, reductions, and pairwise operations
- they are easy to benchmark and easy to keep numerically stable

Suggested kernels:

- rolling statistics
- autocorrelation helpers
- exponential smoothing
- confusion-matrix accumulation
- top-k / ranking helpers

### Batch 2: control and robotics

Target modules:

- `mastermlx.control.lqr`
- `mastermlx.control.mpc`
- `mastermlx.robotics.trajectory`
- `mastermlx.robotics.transforms`

Why:

- these sit on top of already accelerated primitives
- they contain the next most expensive loops
- they benefit from buffer reuse and fewer temporary allocations

Suggested kernels:

- Riccati recursion helpers
- trajectory sampling
- quaternion / transform composition helpers

### Batch 3: estimation and planning

Target modules:

- `mastermlx.estimation`
- `mastermlx.planning` when it lands

Why:

- filtering and search loops are algorithmically stable
- they tend to spend most time in matrix updates and neighbor expansion
- they map well to typed loops in Cython or small C++ kernels

Suggested kernels:

- Kalman predict/update steps
- particle weighting and resampling
- graph expansion and priority queue helpers

## Backend policy

- If a kernel is dense NumPy arithmetic with little branching, prefer Cython.
- If a kernel is shared across many packages and likely to grow, consider C++ with pybind11.
- If a kernel is small but called often, keep it in Python only after profiling proves it is not a bottleneck.

## Release policy

- Every new compiled path must include:
  - fallback implementation
  - unit tests
  - a small benchmark
  - a changelog entry
