# Benchmark Plan

This document defines the first benchmark datasets, the models they should be used with, and the structure for evaluation scripts.

## Principles

- Keep core implementations in `numpy` and `matplotlib`
- Use `scikit-learn` only for dataset loading and baseline comparison
- Prefer a small, fast dataset first, then scale to larger ones
- Record both score and runtime when possible
- Use the same split, seed, and metrics for fair comparison

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
└── benchmark_sequence_models.py
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

