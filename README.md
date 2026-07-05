# mastermlx

`mastermlx` is a NumPy-first machine learning library built from scratch.
It gives you a broad set of classic ML algorithms, math utilities, and optional compiled acceleration in one package.

## Why use it

- Clean top-level API
- 80+ algorithms for classification, regression, clustering, decomposition, NLP, RL, and bandits
- 110+ math tools for metrics, kernels, statistics, distance functions, and time series
- Optional C++ and Cython backends for speed-critical paths
- Pure Python fallback when compiled extensions are not available
- Robotics foundations for transforms, kinematics, trajectories, and Jacobians
- Control foundations for PID, LQR, MPC, and iLQR optimization control
- Optional Cython acceleration for control, robotics, estimation, distance, kernels, particle filters, and time-series hot paths

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

## Highlights

- Models: linear models, trees, ensembles, clustering, decomposition, probabilistic methods, neural nets, SVMs, preprocessing, feature selection
- NLP: vectorizers, tokenizers, vocab builders, language models
- RL and bandits: Q-learning, DQN, REINFORCE, UCB, Thompson sampling, and more
- Math tools: metrics, kernels, distributions, statistical tests, calibration, outlier detection, and time-series helpers

## Benchmarks

The repository includes lightweight benchmark scripts for the main product surfaces:

- `benchmarks/bench_models.py` compares core estimators against scikit-learn baselines
- `benchmarks/bench_accel.py` measures the compiled acceleration layer against NumPy fallbacks
- `benchmarks/bench_tabular.py` exercises the higher-level tabular workflow
- `benchmarks/bench_signal.py` exercises the signal-processing pipeline and streaming helpers

For the benchmark design, dataset choices, and expected output format, see:

- [`docs/benchmark_plan.md`](docs/benchmark_plan.md)
- [`benchmarks/README.md`](benchmarks/README.md)

## Acceleration

The library includes optional compiled helpers for:

- Pairwise distances
- KD-tree search
- Decision tree split search
- Convolution and max pooling

If the compiled backend is missing, `mastermlx` falls back to the NumPy implementation automatically.

## Releases

- Stable releases are published on PyPI: `pip install mastermlx`
- Release tags and changelogs are published on GitHub
- For maintainers, see [`RELEASING.md`](RELEASING.md)

## License

MIT
