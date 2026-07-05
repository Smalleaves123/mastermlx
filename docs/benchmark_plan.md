# Benchmark Plan

This document defines the first benchmark surfaces for `mastermlx`, the datasets or signals they should use, and the structure for evaluation scripts.

The goal is not to build a giant benchmark suite on day one. The goal is to make the current product surfaces easy to demonstrate, easy to compare, and easy to extend.

## Principles

- Keep the core implementation in `numpy` and the project itself
- Use `scikit-learn` only for baselines or well-known dataset helpers when that makes comparison clearer
- Prefer a small, fast dataset first, then scale to larger ones only when the benchmark story needs it
- Record runtime and the task-specific quality metric together when possible
- Use the same split, seed, and metric definition for fair comparison
- Keep scripts deterministic and runnable from the repository root

## Current benchmark surfaces

The first benchmark set should cover the parts of the library that are easiest to explain to users:

1. Core estimators
2. Compiled acceleration
3. Tabular workflows
4. Signal-processing workflows

This gives a more complete story than model-only benchmarks alone.

## Datasets

### 1. Classic small classification sets

- `iris`
- `wine`
- `breast_cancer`

Use these for:
- linear models
- KNN
- decision trees
- random forests
- GaussianNB

### 2. Lightweight image set

- `digits`

Use this for:
- PCA
- KNN
- logistic regression
- decision trees
- random forests
- a first MLP baseline later

### 3. Larger image set

- `MNIST` via `fetch_openml("mnist_784")`

Use this for:
- PCA
- logistic regression
- KNN on a subset
- MLP

### 4. Regression sets

- `diabetes`
- `california_housing`

Use these for:
- linear regression
- ridge regression
- lasso
- tree-based regression later

### 5. Synthetic geometry sets

- `make_blobs`
- `make_moons`

Use these for:
- decision boundary plots
- clustering
- KNN
- tree models
- GMM

### 6. Sequence / discrete sets

- synthetic categorical sequences

Use these for:
- HMM
- n-gram models
- later NLP demos

### 7. Synthetic business workflows

- synthetic tabular classification and regression data
- synthetic audio-like waveforms
- synthetic event streams

Use these for:
- `TabularExperiment`
- `compare_tabular_models`
- `SignalPipeline`
- `StreamingFeatureExtractor`
- `CUSUMDetector`

## Model to dataset map

| Model | Primary datasets | Notes |
|---|---|---|
| LinearRegression | diabetes, california_housing | Report MSE and R2 |
| LogisticRegression | iris, wine, breast_cancer, digits | Report accuracy and loss curve |
| KNNClassifier | iris, wine, digits, make_moons | Show effect of `k` |
| KNNRegressor | diabetes | Compare with linear regression |
| DecisionTreeClassifier | iris, wine, breast_cancer, digits, make_moons | Show tree depth effect |
| RandomForestClassifier | breast_cancer, wine, digits | Compare with single tree |
| PCA | digits, MNIST | Show reconstruction and variance ratio |
| GaussianNB | iris, wine, breast_cancer, digits | Good baseline classifier |
| GMM | make_blobs, digits subsets | Show cluster assignments and log-likelihood |
| HMM | synthetic sequences | Use known hidden states for validation |

## Evaluation metrics

Use these metrics by task:

- Classification:
  - accuracy
  - precision
  - recall
  - F1
  - confusion matrix

- Regression:
  - MSE
  - RMSE
  - R2

- Clustering:
  - silhouette score
  - adjusted rand index when labels are available
  - log-likelihood for GMM

- Dimensionality reduction:
  - explained variance ratio
  - reconstruction error

- Sequence models:
  - log-likelihood
  - decoded path accuracy on synthetic data

## Evaluation script structure

Suggested layout:

```text
scripts/
├── benchmark_classification.py
├── benchmark_regression.py
├── benchmark_clustering.py
├── benchmark_dimensionality_reduction.py
├── benchmark_sequence_models.py
├── benchmark_tabular.py
└── benchmark_signal.py
```

Each script should follow the same flow:

1. Load dataset
2. Split data
3. Fit core model
4. Fit baseline model from scikit-learn
5. Compute metrics
6. Measure runtime
7. Save results to `outputs/` or print a summary table

## Recommended output format

Each benchmark run should produce:

- a small table of metrics
- one or more figures
- a plain-text summary

For the current repository, plain-text summaries are enough for the first pass. CSV or JSON export can be added later once the benchmark set stabilizes.

Example files:

```text
outputs/
├── classification_iris.csv
├── regression_diabetes.csv
├── digits_pca.png
└── gmm_blobs.png
```

## First benchmark batch

Start with these because they are fast and high value:

1. `iris` with logistic regression, KNN, GaussianNB, decision tree
2. `wine` with GaussianNB and decision tree
3. `breast_cancer` with logistic regression, random forest
4. `digits` with PCA, logistic regression, KNN
5. `diabetes` with linear regression
6. `make_blobs` with GMM
7. synthetic sequences with HMM
8. synthetic tabular classification with `TabularExperiment`
9. synthetic waveform and event-stream checks with `SignalPipeline` and `CUSUMDetector`

## Current scripts

The repository already contains the first executable scripts for these surfaces:

- [`benchmarks/bench_models.py`](../benchmarks/bench_models.py)
- [`benchmarks/bench_accel.py`](../benchmarks/bench_accel.py)
- [`benchmarks/bench_tabular.py`](../benchmarks/bench_tabular.py)
- [`benchmarks/bench_signal.py`](../benchmarks/bench_signal.py)

These scripts are intentionally lightweight so they can be used during development without becoming a maintenance burden.
