# mastermlx Technical Overview

`mastermlx` is a NumPy-based machine learning library organized as a single top-level namespace.
Most models are available directly from `import mastermlx as mlx`.

## Core Design

- NumPy-first implementation
- Pure Python APIs with optional compiled acceleration
- Broad algorithm coverage across classical ML, neural nets, NLP, RL, bandits, and signal processing
- Consistent estimator-style interfaces: `fit`, `predict`, `transform`, `score`

## Installation

```bash
pip install mastermlx
```

Development setup:

```bash
pip install -e ".[dev,compare]"
```

## Top-Level Usage

```python
import numpy as np
import mastermlx as mlx

X = np.random.randn(100, 4)
y = np.random.randint(0, 2, size=100)

clf = mlx.LogisticRegression().fit(X, y)
pred = clf.predict(X)

scaler = mlx.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

from mastermlx import entropy, pairwise_distance, silhouette
```

### Tabular data checks

```python
from mastermlx import AutoPreprocessor, quality_report

report = quality_report(X)
prep = AutoPreprocessor().fit(X)
X_ready = prep.transform(X)
feature_names = prep.get_feature_names_out()
```

`quality_report` summarizes missing values, duplicates, constant columns,
categorical frequencies, numeric outliers, and target quality. Use
`compare_schema` or `drift_report` for train/test checks. `AutoPreprocessor`
detects numeric and categorical columns, imputes missing values, scales numeric
features, one-hot encodes categorical features, and ignores unseen categories
by default.

## Module Map

### `linear_models`

- `LinearRegression`
- `LogisticRegression`
- `RidgeRegression`
- `LassoRegression`
- `ElasticNetRegression`
- `SGDClassifier`
- `SGDRegressor`
- `Perceptron`
- `HuberRegressor`

### `trees`

- `DecisionTreeClassifier`
- `DecisionTreeRegressor`
- `RandomForestClassifier`
- `RandomForestRegressor`
- `GradientBoostingClassifier`
- `GradientBoostingRegressor`
- `AdaBoostClassifier`
- `AdaBoostRegressor`

### `ensemble`

- `BaggingClassifier`
- `BaggingRegressor`
- `ExtraTreesClassifier`
- `ExtraTreesRegressor`
- `StackingClassifier`
- `StackingRegressor`
- `VotingClassifier`
- `VotingRegressor`

### `clustering`

- `KMeans`
- `DBSCAN`
- `GMM`
- `MeanShift`
- `AffinityPropagation`
- `AgglomerativeClustering`
- `SpectralClustering`

### `decomposition`

- `PCA`
- `KernelPCA`
- `NMF`
- `ICA`
- `FastICA`
- `TruncatedSVD`
- `FactorAnalysis`

### `probabilistic`

- `GaussianNB`
- `BernoulliNB`
- `MultinomialNB`
- `LDA`
- `QDA`
- `HMM`
- `GaussianProcessRegressor`

### `neural_net`

- `Sequential`
- `MLPClassifier`
- `MLPRegressor`
- layers such as `Dense`, `Dropout`, `BatchNorm`, `Conv2D`, `MaxPool2D`, `AttentionPooling1D`

### `preprocessing`

- `StandardScaler`
- `MinMaxScaler`
- `RobustScaler`
- `Normalizer`
- `OneHotEncoder`
- `OrdinalEncoder`
- `LabelEncoder`
- `TargetEncoder`
- `SimpleImputer`
- `PolynomialFeatures`
- `Pipeline`

### `neighbors`

- `KNNClassifier`
- `KNNRegressor`
- `RadiusNeighborsClassifier`
- `RadiusNeighborsRegressor`
- `CentroidClassifier`

### `nlp`

- `CountVectorizer`
- `TfidfVectorizer`
- `HashingVectorizer`
- `SimpleTokenizer`
- `CharTokenizer`
- `Vocab`
- `NGramLanguageModel`

### `rl`

- `QLearningAgent`
- `DQNAgent`
- `REINFORCEAgent`
- `SarsaAgent`
- `DoubleQAgent`

### `robotics`

- `DHLink`
- `forward_kinematics`
- `inverse_kinematics`
- `geometric_jacobian`
- `cubic_time_scaling`
- `quintic_time_scaling`
- `joint_trajectory`
- `plot_chain`
- `plan_joint_path`
- `smooth_joint_path`
- `plan_joint_trajectory`
- `PlanarPoseEKF`
- state-estimation helpers for odometry, position, and heading fusion

### `estimation`

- `KalmanFilter`
- `ExtendedKalmanFilter`
- `ParticleFilter`
- `systematic_resample`

### `control`

- `PIDController`
- `DiscreteLQR`
- `finite_horizon_lqr`
- `LinearMPC`
- `iLQR`

### `bandits`

- `EpsilonGreedyBandit`
- `UCBBandit`
- `ThompsonSampling`
- `SoftmaxBandit`
- `LinUCBBandit`
- `Exp3Bandit`

## Acceleration Layer

Compiled helpers live under `mastermlx.accel` and are used automatically when available.

Current accelerated areas:

- Distance computations
- KD-tree nearest-neighbor search
- Decision tree split search
- Convolution and pooling helpers

The package falls back to pure NumPy implementations if a backend cannot be loaded.

## Testing

```bash
pytest tests/
```

## Notes for Contributors

- Keep the top-level API stable
- Preserve fallback behavior when adding new compiled paths
- Add tests for both compiled and pure-Python execution paths when practical
