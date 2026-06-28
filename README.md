# mastermlx

A from-scratch machine learning library — 80+ algorithms, 110+ math tools, pure NumPy.

```python
import mastermlx as mlx

# All models at top level
model = mlx.RandomForestClassifier(n_estimators=100).fit(X, y)
print(model.score(X, y))

# All math tools in one place
from mastermlx import entropy, cosine_sim, silhouette, smote
```

## Install

```bash
pip install mastermlx
# or editable:
pip install -e ".[dev,compare]"
```

Dependencies: `numpy`, `matplotlib`, `seaborn`, `plotly`.  SciPy / scikit-learn are optional extras.

## Models

| Module | Algorithms |
|--------|-----------|
| `linear_models` | `LinearRegression`, `LogisticRegression`, `RidgeRegression`, `LassoRegression`, `ElasticNetRegression`, `SGDClassifier`, `SGDRegressor`, `Perceptron`, `HuberRegressor` |
| `neural_net` | `Sequential`, `MLPClassifier`, `MLPRegressor`, Dense/Dropout/BatchNorm/Attention layers, SGD/Adam/RMSProp |
| `svm` | `SVC`, `LinearSVR`, `KernelSVR`, `OneClassSVM` |
| `trees` | `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor`, `GradientBoostingClassifier`, `GradientBoostingRegressor`, `AdaBoostClassifier`, `AdaBoostRegressor` |
| `ensemble` | `BaggingClassifier`, `BaggingRegressor`, `ExtraTreesClassifier`, `ExtraTreesRegressor`, `StackingClassifier`, `StackingRegressor`, `VotingClassifier`, `VotingRegressor` |
| `clustering` | `KMeans`, `DBSCAN`, `GMM`, `MeanShift`, `AffinityPropagation`, `AgglomerativeClustering`, `SpectralClustering`, Bayesian/Variational GMM |
| `decomposition` | `PCA`, `KernelPCA`, `NMF`, `ICA`, `FastICA`, `TruncatedSVD`, `FactorAnalysis` |
| `manifold` | `Isomap`, `LLE`, `MDS`, `SpectralEmbedding` |
| `probabilistic` | `GaussianNB`, `BernoulliNB`, `MultinomialNB`, `LDA`, `QDA`, `HMM`, `GaussianProcessRegressor`, `BayesianLinearRegression` |
| `variational` | `VariationalLinearRegression`, `VariationalLogisticRegression`, `VariationalPoissonRegression`, `VariationalGaussianMixture`, `BayesianGaussianMixture` |
| `anomaly` | `IsolationForest`, `LocalOutlierFactor`, `HBOS`, `EllipticEnvelope` |
| `preprocessing` | `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `Normalizer`, `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, `TargetEncoder`, `Binarizer`, `KBinsDiscretizer`, `PolynomialFeatures`, `PowerTransform`, `QuantileTransform`, `SimpleImputer`, `Pipeline` |
| `data` | `train_test_split`, `KFold`, `StratifiedKFold`, `GroupKFold`, `TimeSeriesSplit`, `ShuffleSplit`, `GridSearchCV`, `RandomizedSearchCV`, `cross_val_score`, `learning_curve` |
| `selection` | `SelectKBest`, `RFE`, `VarianceThreshold`, `f_classif`, `f_regression` |
| `nlp` | `CountVectorizer`, `TfidfVectorizer`, `HashingVectorizer`, `NGramLanguageModel`, `SimpleTokenizer`, `CharTokenizer`, `Vocab`, `TextSeq`, `SparseCOO` |
| `signal` | `mfcc`, `mel_spectrogram`, `stft`, `istft`, `bandpass_filter`, `convolve1d`, `frame_signal`, `hamming_window`, `hann_window` |
| `rl` | `QLearningAgent` |
| `bandits` | `EpsilonGreedyBandit`, `UCBBandit`, `ThompsonSampling`, `SoftmaxBandit`, `LinUCBBandit`, `Exp3Bandit` |
| `semi_supervised` | `LabelPropagation`, `LabelSpreading` |
| `time_series` | `ARModel`, `dtw_distance`, `dtw_path`, `autocorrelation`, `partial_autocorrelation`, `exponential_smoothing`, `cusum_change_points` |

## Math Tools (`mastermlx.math_tools`)

116 functions, zero extra dependencies.

### Distance / Similarity
`euclidean_distance`, `manhattan_distance`, `minkowski_distance`, `chebyshev_distance`, `cosine_distance`, `hamming_distance`, `jaccard_distance`, `mahalanobis_distance`, `pairwise_distance`, `cosine_sim`, `dot_sim`, `pearson_r`, `spearman_r`, `kendall_tau`, `pairwise_cosine`

### Kernels
`linear_kernel`, `cosine_kernel`, `poly_kernel`, `rbf_kernel`, `laplacian_kernel`, `sigmoid_kernel`, `chi2_kernel`, `additive_chi2_kernel`, `hellinger_kernel`, `pairwise_kernel`

### Metrics
`accuracy`, `f1_score`, `precision_score`, `recall_score`, `roc_auc_score`, `log_loss`, `r2_score`, `mae`, `mse`, `rmse`, `mape`, `explained_variance_score`, `confusion_matrix`, `balanced_accuracy_score`, `cohen_kappa_score`, `matthews_corrcoef`, `hinge_loss`, `specificity_score`, `top_k_accuracy_score`, `zero_one_loss`

### Information Theory
`entropy`, `cross_entropy`, `kl_divergence`, `js_divergence`, `mutual_information`, `joint_entropy`, `conditional_entropy`, `variation_of_information`, `normalized_mutual_information`, `empirical_distribution`

### Statistical Tests
`mann_whitney`, `wilcoxon`, `kruskal`, `chi2_contingency`, `f_oneway`, `ks_test`

### Distributions
`Normal`, `Uniform`, `Exponential`, `LogNormal`, `Chi2`, `StudentT`, `Poisson`, `Bernoulli`

### Noise / Augmentation
`gauss`, `uniform`, `laplace`, `salt_pepper`, `dropout`, `poisson`, `jitter`, `shuffle`, `swap`, `mixup`, `cutmix`, `smote`

### Clustering / Outliers / Calibration
`silhouette`, `davies_bouldin`, `calinski_harabasz`, `adj_rand`, `adj_mi`, `v_measure`, `zscore`, `mod_zscore`, `iqr_outliers`, `grubbs`, `brier_score`, `reliability_curve`, `expected_calibration_error`

### Time Series / Attention
`ARModel`, `dtw_distance`, `dtw_path`, `autocorrelation`, `partial_autocorrelation`, `exponential_smoothing`, `cusum_change_points`, `rolling_mean`, `difference`, `scaled_dot_product_attention`, `multi_head_attention`, `sinusoidal_positional_encoding`

## Quickstart

```python
import numpy as np
import mastermlx as mlx

# --- Classification ---
X, y = np.random.randn(200, 5), np.where(np.random.randn(200) > 0, 1, 0)
clf = mlx.SGDClassifier(loss="hinge", max_iter=50).fit(X, y)
print("acc:", clf.score(X, y))

# --- Clustering ---
kmeans = mlx.KMeans(n_clusters=3, random_state=0).fit(X)
print("inertia:", kmeans.inertia_)

# --- Preprocessing ---
X_scaled = mlx.StandardScaler().fit_transform(X)

# --- Math tools ---
from mastermlx import entropy, pearson_r, silhouette
print("silhouette:", silhouette(X, kmeans.labels_))
print("entropy:", entropy(np.array([0.2, 0.3, 0.5])))
```

## Layout

```
mastermlx/
  math_tools/     ← 116 utility functions (distance, kernels, stats, ...)
  linear_models/  ← linear & regularized models, SGD, perceptron
  neural_net/     ← Sequential, MLP, layers, optimizers
  trees/          ← decision trees, random forest, gradient boosting
  svm/            ← SVC, SVR, one-class SVM
  ensemble/       ← bagging, stacking, voting, extra trees
  clustering/     ← k-means, DBSCAN, GMM, spectral, ...
  decomposition/  ← PCA, NMF, ICA, kernel PCA, ...
  preprocessing/  ← scalers, encoders, imputers, pipeline
  probabilistic/  ← naive bayes, HMM, Gaussian process, ...
  variational/    ← variational Bayesian inference
  anomaly/        ← isolation forest, LOF, HBOS
  nlp/            ← tokenizers, vectorizers, language model
  signal/         ← STFT, MFCC, filters, convolutions
  data/           ← CV splitters, grid search, model selection
  selection/      ← feature selection (RFE, SelectKBest)
  bandits/        ← multi-armed bandits
  rl/             ← Q-learning
  semi_supervised/← label propagation
  viz/            ← plotting utilities
tests/            ← 327 tests, all passing
```

## Dev

```bash
conda activate cv1                        # or your Python >=3.10 env
pip install -e ".[dev,compare]"
pytest tests/                              # 327 passed
```

## License

MIT
