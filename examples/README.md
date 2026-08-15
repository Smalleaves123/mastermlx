# mastermlx 0.1.15 Examples and Tutorials

This directory is the copy-and-run learning path for `mastermlx` 0.1.15. It
covers the common estimator contract, supervised and unsupervised learning,
NLP, probabilistic models, bandits, reinforcement learning, signal processing,
and robotics workflows.

For a task-oriented list of public classes and functions, start with the
[`0.1.15 API guide`](API_REFERENCE.md).

## Installation

Install the released package for examples that do not create plots:

```bash
python -m pip install "mastermlx==0.1.15"
```

Install plotting dependencies as well:

```bash
python -m pip install "mastermlx[viz]==0.1.15"
```

When working from a source checkout, run this from the repository root so the
examples use the local code:

```bash
python -m pip install -e ".[viz]"
```

The project development environment is the conda environment named `CV`. A
source-checkout example can therefore be run as:

```bash
conda run -n CV python examples/quickstart/basic_ml.py
```

## The common estimator interface

Most tabular models follow a small, scikit-learn-style contract:

```python
from mastermlx.linear_models import LogisticRegression

model = LogisticRegression(lr=0.1, n_iter=200)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
quality = model.score(X_test, y_test)
parameters = model.get_params()
```

- `fit(X, y)` learns state and returns the fitted object.
- `predict(X)` returns one output per input sample.
- `score(X, y)` is accuracy for classifiers and R² for regressors.
- `get_params()` and `set_params(...)` expose constructor parameters.
- fitted attributes end in `_`, for example `coef_`, `classes_`, or
  `n_features_in_`.
- transformers use `fit`, `transform`, and `fit_transform`.

`X` is always a two-dimensional matrix with shape
`(n_samples, n_features)`. Keep the sample axis when predicting one item:

```python
one_prediction = model.predict(X_test[:1])  # shape: (1,)
```

Pipelines chain transformers and a final estimator:

```python
from mastermlx.linear_models import RidgeRegression
from mastermlx.preprocessing import Pipeline, StandardScaler

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", RidgeRegression(alpha=1.0)),
])
pipeline.fit(X_train, y_train)
pipeline.set_params(model__alpha=0.5)
```

See [`regression/regression_pipeline.py`](regression/regression_pipeline.py)
for a complete split, fit, predict, and evaluation workflow.

## Tutorial map

| Area | Start here | What it demonstrates | Plotting |
| --- | --- | --- | --- |
| Quick start | [`quickstart/basic_ml.py`](quickstart/basic_ml.py) | Regression, PCA, and K-means | No |
| Classification | [`classification/compare_models.py`](classification/compare_models.py) | Four classifiers and confusion matrices | Yes |
| Regression | [`regression/regression_pipeline.py`](regression/regression_pipeline.py) | Split, pipeline, metrics, nested parameters | No |
| Clustering | [`clustering/kmeans_demo.py`](clustering/kmeans_demo.py) | K-means, inertia, silhouette | Yes |
| Neural networks | [`neural_networks/mlp_spirals.py`](neural_networks/mlp_spirals.py) | MLP training and decision regions | Yes |
| NLP classification | [`nlp/text_classify.py`](nlp/text_classify.py) | TF-IDF and logistic regression | Yes |
| NLP topics | [`nlp/topic_modeling.py`](nlp/topic_modeling.py) | Count vectors and variational LDA | No |
| Probabilistic ML | [`probabilistic/probabilistic_models.py`](probabilistic/probabilistic_models.py) | GP uncertainty and discriminant LDA | No |
| Bandits | [`bandits/bandit_comparison.py`](bandits/bandit_comparison.py) | Epsilon-greedy, UCB, Thompson sampling | No |
| Reinforcement learning | [`rl/q_learning_demo.py`](rl/q_learning_demo.py) | GridWorld training and greedy evaluation | No |
| Tabular readiness | [`tabular/readiness_demo.py`](tabular/readiness_demo.py) | Contracts, quality, and drift checks | No |
| Signal processing | [`signal/README.md`](signal/README.md) | Fourier, streaming, monitoring, health | Mixed |
| Robotics | [`robotics/README.md`](robotics/README.md) | Kinematics, planning, simulation, workcells | Mixed |
| Math tools | [`tools/math_tools_demo.py`](tools/math_tools_demo.py) | Statistics, kernels, and diagnostics | Yes |

Plotting scripts write PNG files below `examples/outputs/`. Keep the output
directory in place; generated PNG files are safe to remove and recreate.

## Recommended learning paths

For a first tour of the estimator API:

1. Run `quickstart/basic_ml.py`.
2. Follow `regression/regression_pipeline.py` to learn splitting and pipelines.
3. Choose `classification/compare_models.py` or `clustering/kmeans_demo.py`.
4. Use the [API guide](API_REFERENCE.md) to select the next model.

For sequential decision making:

1. Run `bandits/bandit_comparison.py` to learn `select_arm` and `update`.
2. Run `rl/q_learning_demo.py` to learn environment, agent, training, and
   evaluation interfaces.

For robotics, use the ordered path in [`robotics/README.md`](robotics/README.md).
The standard environment contract and lightweight-physics boundary are in
[`docs/robotics_simulation.md`](../docs/robotics_simulation.md).

## Import guidance

Prefer domain imports in application code because they make intent explicit:

```python
from mastermlx.clustering import KMeans
from mastermlx.preprocessing import StandardScaler
```

The broad top-level namespace remains convenient for exploration:

```python
import mastermlx as mlx

model = mlx.KMeans(n_clusters=3, random_state=0)
```

`LDA` is intentionally not a top-level name because it has two unrelated
meanings. Import one of the explicit aliases:

```python
from mastermlx.nlp import NLP_LDA                     # topic model
from mastermlx.probabilistic import DiscriminantLDA  # classifier
```

## Reproducibility

Examples use fixed `random_state` values or `numpy.random.default_rng(seed)`.
When adapting them, pass a seed to stochastic estimators rather than modifying
NumPy's global random state.
