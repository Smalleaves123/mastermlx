# mastermlx

`mastermlx` is a NumPy-first machine learning library built from scratch.
It gives you a broad set of classic ML algorithms, math utilities, and optional compiled acceleration in one package.

## Why use it

- Clean top-level API
- 80+ algorithms for classification, regression, clustering, decomposition, NLP, RL, and bandits
- 110+ math tools for metrics, kernels, statistics, distance functions, and time series
- Data quality, schema, drift, and explicit tabular data-contract checks
- Unified OOF/CV evaluation reports with bootstrap uncertainty and learning curves
- Online tabular learning with incremental updates, sliding windows, drift alerts, and delayed labels
- Optional C++ and Cython backends for speed-critical paths
- Pure Python fallback when compiled extensions are not available
- Robotics foundations for transforms, kinematics, trajectories, Jacobians, canonical model aliases, and batch evaluation
- General serial URDF spatial kinematics with arbitrary joint axes, batched FK/Jacobians, and damped task-space IK
- 3D sphere/box/capsule collision checks and NumPy-only voxel occupancy maps from point clouds
- URDF collision boxes, primitives, OBJ/STL meshes, and spatial RRT/RRT* joint planning
- Constraint-aware trajectory optimization with velocity, acceleration, and jerk limits
- Spatial URDF dynamics with mass matrix, gravity, Coriolis, inverse/forward dynamics, and computed torque
- Semi-supervised learning with graph label propagation, label spreading, and inductive self-training
- Joint-path optimization with curvature, reference-path, joint-limit, and workcell clearance costs
- Optional MPC trajectory tracking in the virtual joint-space simulator
- Planar workcell workflow with configurable joint limits, Cartesian task interpolation, pick-and-place cycles with gripper events, clearance-aware paths, constrained retiming, kinematic diagnostics, virtual tracking, and CSV/JSON exports
- Control foundations for PID, LQR, MPC, and iLQR optimization control
- Optional C++/Cython acceleration for control, robotics, estimation, distance, kernels, particle filters, and time-series hot paths

## Install

```bash
pip install mastermlx
```

If you want the latest code from GitHub:

```bash
pip install git+https://github.com/Smalleaves123/mastermlx.git
```

For development:

```bash
pip install -e ".[dev,compare]"
```

For plotting helpers and visual examples, install the optional visualization
extra:

```bash
pip install "mastermlx[viz]"
```

The core package depends only on NumPy; plotting and comparison baselines are
kept as optional extras.

## Quick Example

```python
import numpy as np
import mastermlx as mlx

X = np.random.randn(200, 5)
y = np.where(np.random.randn(200) > 0, 1, 0)

clf = mlx.SGDClassifier(loss="hinge", max_iter=50).fit(X, y)
print(clf.score(X, y))

kmeans = mlx.KMeans(n_clusters=3, random_state=0).fit(X)
print(kmeans.inertia_)

print(mlx.entropy(np.array([0.2, 0.3, 0.5])))
```

For copy-and-run tutorials covering robotics and core ML APIs, see the
[public examples](examples/README.md).

## Highlights

- Models: linear models, trees, ensembles, clustering, decomposition, probabilistic methods, neural nets, SVMs, preprocessing, feature selection
- NLP: vectorizers, tokenizers, vocab builders, language models
- RL and bandits: Q-learning, DQN, REINFORCE, UCB, Thompson sampling, and more
- Math tools: metrics, kernels, distributions, statistical tests, calibration, outlier detection, and time-series helpers

## Business workflows

- `DataReadinessReport` for schema, quality, drift, and contract checks
- `SignalHealthExperiment` for sensor health scoring and vibration monitoring
- `RobotWorkcell.plan_motion()` and `RobotWorkcell.plan_pick_and_place()` for collision-aware motion, task cycles, retiming, tracking, and safety reports

For the recommended use cases and workflow conventions, see:

- [`docs/business_use_cases.md`](docs/business_use_cases.md)
- [`docs/workflows.md`](docs/workflows.md)
- [`docs/api_policy.md`](docs/api_policy.md)
- [`docs/semi_supervised.md`](docs/semi_supervised.md)

## Benchmarks

The repository includes lightweight benchmark scripts for the main product surfaces:

- `benchmarks/bench_models.py` compares core estimators against scikit-learn baselines
- `benchmarks/bench_ml_comparison.py` compares ML estimators with scikit-learn and SciPy
- `benchmarks/bench_accel.py` measures the compiled acceleration layer against NumPy fallbacks
- `benchmarks/bench_tabular.py` exercises the higher-level tabular workflow
- `benchmarks/bench_signal.py` exercises the signal-processing pipeline and streaming helpers

For the benchmark design, dataset choices, and expected output format, see:

- [`docs/benchmark_plan.md`](docs/benchmark_plan.md)
- [`benchmarks/README.md`](benchmarks/README.md)

## Neural API and persistence

The neural estimator shapes, `evaluate()` result format, safe versioned
checkpoints, and backend selection rules are documented in
[`docs/neural_api.md`](docs/neural_api.md).

## Acceleration

The library includes optional compiled helpers for:

- Pairwise distances
- KD-tree search
- Decision tree split search
- Convolution and max pooling

If the compiled backend is missing, `mastermlx` falls back to the NumPy implementation automatically.
Source builds can explicitly disable native extensions with
`MASTERML_DISABLE_EXTENSIONS=1` when no compiler is available.

## Releases

- Stable releases are published on PyPI: `pip install mastermlx`
- Release tags and changelogs are published on GitHub
- For maintainers, see [`RELEASING.md`](RELEASING.md)

## License

MIT
