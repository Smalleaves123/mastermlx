# mastermlx 0.1.15 API Guide

This is a curated, task-oriented guide to the public API used by the examples.
It is not an inventory of every compatibility alias. Public names are defined
by each package's `__all__`; prefer the domain import paths below for readable,
forward-compatible code.

## Core conventions

| Object type | Training | Inference | Evaluation |
| --- | --- | --- | --- |
| Classifier | `fit(X, y)` | `predict(X)`, sometimes `predict_proba(X)` | `score(X, y)` returns accuracy |
| Regressor | `fit(X, y)` | `predict(X)` | `score(X, y)` returns R² |
| Transformer | `fit(X[, y])` | `transform(X)`, `fit_transform(X[, y])` | task-specific |
| Clusterer | `fit(X)` | `predict(X)` when supported | attributes such as `labels_`, `inertia_` |
| Experiment | task-specific `fit` or `run` | report/result object | report fields and exported artifacts |

Array contracts in 0.1.15:

- `X`: two-dimensional, `(n_samples, n_features)`.
- `y`: normally one-dimensional, `(n_samples,)`.
- classifier and regressor `predict(X)`: preserves the sample axis, including
  one-sample input.
- `predict_proba(X)`: `(n_samples, n_classes)`.
- incompatible feature counts raise an error after fitting.
- learned attributes use a trailing underscore.

Estimators derived from `BaseEstimator` expose `get_params`, `set_params`,
`state_dict`, `load_state_dict`, `save`, and `load`. Transformers expose the
parameter methods and the transform contract. The safe checkpoint format is
versioned and does not use pickle.

## Data, validation, and composition

```python
from mastermlx.data import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from mastermlx.preprocessing import Pipeline, StandardScaler
from mastermlx.utils import clone
```

| Interface | Purpose |
| --- | --- |
| `train_test_split` | Reproducible train/test arrays |
| `KFold`, `StratifiedKFold`, `TimeSeriesSplit` | Cross-validation iterators |
| `cross_val_score`, `cross_validate`, `cross_val_predict` | Model evaluation |
| `GridSearchCV`, `RandomizedSearchCV` | Hyperparameter search |
| `Pipeline` | Ordered transforms plus a final estimator |
| `clone` | Unfitted estimator copy with the same parameters |

Pipeline nested parameters use `step__parameter`, for example
`pipeline.set_params(model__alpha=0.1)`.

## Preprocessing and feature selection

```python
from mastermlx.preprocessing import (
    ColumnTransformer,
    OneHotEncoder,
    Pipeline,
    SimpleImputer,
    StandardScaler,
)
from mastermlx.selection import SelectKBest, VarianceThreshold
```

Main preprocessing interfaces include `StandardScaler`, `MinMaxScaler`,
`RobustScaler`, `Normalizer`, `SimpleImputer`, `KNNImputer`, `OneHotEncoder`,
`OrdinalEncoder`, `LabelEncoder`, `PolynomialFeatures`, `KBinsDiscretizer`,
`PowerTransform`, `QuantileTransform`, `ColumnTransformer`, and
`AutoPreprocessor`.

Feature selection includes `VarianceThreshold`, `SelectKBest`, `RFE`,
`SequentialFeatureSelector`, and `SelectFromModel`.

## Classification

```python
from mastermlx.linear_models import LogisticRegression, SGDClassifier
from mastermlx.neighbors import KNNClassifier
from mastermlx.svm import SVC
from mastermlx.trees import RandomForestClassifier
```

| Family | Main public interfaces |
| --- | --- |
| Linear | `LogisticRegression`, `SGDClassifier`, `Perceptron` |
| Neighbors | `KNNClassifier`, `RadiusNeighborsClassifier`, `NearestCentroid` |
| Trees | `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `AdaBoostClassifier` |
| Ensembles | `BaggingClassifier`, `ExtraTreesClassifier`, `VotingClassifier`, `StackingClassifier`, `CalibratedClassifierCV` |
| SVM | `SVC`, `NuSVC` |
| Probabilistic | `GaussianNB`, `BernoulliNB`, `MultinomialNB`, `DiscriminantLDA`, `QDA` |
| Neural | `MLPClassifier`, `Sequential` |

The complete comparison tutorial is
[`classification/compare_models.py`](classification/compare_models.py).

## Regression

```python
from mastermlx.linear_models import LinearRegression, RidgeRegression
from mastermlx.trees import RandomForestRegressor
```

| Family | Main public interfaces |
| --- | --- |
| Linear and robust | `LinearRegression`, `RidgeRegression`, `LassoRegression`, `ElasticNetRegression`, `HuberRegressor`, `RANSACRegressor`, `QuantileRegressor`, `SGDRegressor` |
| Neighbors | `KNNRegressor`, `RadiusNeighborsRegressor` |
| Trees and ensembles | `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`, `AdaBoostRegressor`, `ExtraTreesRegressor`, `HistGradientBoostingRegressor` |
| SVM | `LinearSVR`, `KernelSVR` |
| Probabilistic | `BayesianLinearRegression`, `GaussianProcessRegressor`, `VariationalLinearRegression` |
| Neural | `MLPRegressor`, `Sequential` |

All listed regressor `score` methods use R². See
[`regression/regression_pipeline.py`](regression/regression_pipeline.py).

## Clustering and representation learning

```python
from mastermlx.clustering import DBSCAN, GMM, KMeans
from mastermlx.decomposition import NMF, PCA, TruncatedSVD
```

Clustering interfaces include `KMeans`, `MiniBatchKMeans`, `DBSCAN`, `GMM`,
`BayesianGaussianMixture`, `AgglomerativeClustering`, `SpectralClustering`,
`AffinityPropagation`, and `MeanShift`.

Representation learning includes `PCA`, `KernelPCA`, `TruncatedSVD`, `NMF`,
`FastICA`, `FactorAnalysis`, `CCA`, and `NCA`; the `mastermlx.manifold` package
provides nonlinear embedding methods.

See [`clustering/kmeans_demo.py`](clustering/kmeans_demo.py).

## NLP

```python
from mastermlx.nlp import CountVectorizer, NLP_LDA, TfidfVectorizer
```

| Interface | Main methods or outputs |
| --- | --- |
| `CountVectorizer` | `fit_transform(texts)`, `transform(texts)`, `feature_names_` |
| `TfidfVectorizer` | Count-vectorizer contract plus `idf_` |
| `HashingVectorizer` | Fixed-width stateless text features |
| `SimpleTokenizer`, `CharTokenizer` | Token sequences |
| `Vocab`, `TextSeq`, `SeqPad` | Vocabulary and padded sequence preparation |
| `NGramLanguageModel` | N-gram fitting, scoring, and generation |
| `NLP_LDA` | `fit_transform(counts)`, `transform(counts)`, `components_`, `perplexity(counts)` |

`mastermlx.LDA` deliberately raises `AttributeError`: use `NLP_LDA` for latent
Dirichlet allocation and `DiscriminantLDA` for linear discriminant analysis.
See [`nlp/text_classify.py`](nlp/text_classify.py) and
[`nlp/topic_modeling.py`](nlp/topic_modeling.py).

## Probabilistic models

```python
from mastermlx.probabilistic import (
    BayesianLinearRegression,
    DiscriminantLDA,
    GaussianProcessRegressor,
)
```

The package includes naive Bayes classifiers, LDA/QDA classifiers, Bayesian
linear regression, Gaussian-process regression, kernel density estimation,
hidden Markov models, common distributions, and variational models.

`BayesianLinearRegression.predict(X, return_std=True)` and
`GaussianProcessRegressor.predict(X, return_std=True)` return `(mean, std)`.
See the runnable
[`probabilistic tutorial`](probabilistic/probabilistic_models.py).

## Neural networks

```python
from mastermlx.neural_net import Adam, Dense, MLPClassifier, ReLU, Sequential
```

High-level estimators are `MLPClassifier`, `MLPRegressor`, and `Sequential`.
Layers include dense, convolutional, pooling, normalization, recurrent,
embedding, and attention layers. Optimizers include `SGD`, `Adam`, `AdamW`,
`RMSProp`, and `AdaGrad`; callbacks and learning-rate schedulers are public.

See [`neural_networks/mlp_spirals.py`](neural_networks/mlp_spirals.py) and the
full [`neural API contract`](../docs/neural_api.md).

## Bandits and reinforcement learning

```python
from mastermlx.bandits import EpsilonGreedyBandit, UCBBandit
from mastermlx.rl import GridWorld, QLearningAgent, evaluate, train_tabular
```

Bandit objects use `select_arm()` followed by `update(arm, reward)`. Available
policies include epsilon-greedy, softmax, UCB, EXP3, Bernoulli Thompson
sampling, LinUCB, and linear Thompson sampling.

RL environments use `reset()` and `step(action)`. Agents use `select_action`
and `update`; `train_tabular` and `evaluate` provide the standard loop for
`QLearningAgent`, `SARSAAgent`, and `DoubleQLearningAgent`. `DQNAgent` and
`REINFORCEAgent` cover neural policies.

See the [bandit tutorial](bandits/bandit_comparison.py) and
[Q-learning tutorial](rl/q_learning_demo.py).

## Anomaly detection, signal, and time series

```python
from mastermlx.anomaly import IsolationForest
from mastermlx.signal import SignalExperiment
```

Anomaly estimators follow the 0.1.15 sample-axis contract for `score_samples`
and `decision_function`. Signal APIs cover FFT/STFT, filtering, streaming,
event detection, multi-channel alignment, features, monitoring, and workflow
reports. Time-series utilities are exported from `mastermlx.math_tools`.

Start with [`signal/fourier_demo.py`](signal/fourier_demo.py), then use the
ordered list in [`signal/README.md`](signal/README.md).

## Robotics, planning, control, and simulation

The robotics surface is intentionally documented by workflow because model,
trajectory, collision, planning, estimation, control, and simulation objects
work together. Start with [`robotics/00_quickstart.py`](robotics/00_quickstart.py)
and follow [`robotics/README.md`](robotics/README.md).

Detailed contracts:

- [`3D robotics and URDF API`](../docs/robotics_3d.md)
- [`simulation environment contract`](../docs/robotics_simulation.md)
- [`control API`](../docs/control.md)
- [`workflow result objects`](../docs/workflows.md)

## Metrics and mathematical tools

```python
from mastermlx.utils import accuracy, confusion_matrix, r2_score, rmse
from mastermlx.math_tools import entropy, rbf_kernel, silhouette
```

`mastermlx.utils` contains validation, estimator utilities, common metrics,
distances, kernels, gradients, and random-state helpers. `mastermlx.math_tools`
adds clustering metrics, information theory, distributions, statistical tests,
calibration, time-series helpers, noise and augmentation, and toy dataset
generators.

## Backend selection

```python
import mastermlx as mlx

mlx.set_backend("auto")    # compiled implementation when available
mlx.set_backend("numpy")   # deterministic NumPy fallback
print(mlx.get_backend())
```

The public behavior should be the same across backends. See the
[`acceleration guide`](../docs/acceleration.md) for build and diagnostic
details.
