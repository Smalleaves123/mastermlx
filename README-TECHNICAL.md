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
from mastermlx import AutoPreprocessor, DataContract, EvaluationReport, quality_report

report = quality_report(X)
prep = AutoPreprocessor().fit(X)
X_ready = prep.transform(X)
feature_names = prep.get_feature_names_out()

contract = DataContract(
    rules={"age": {"kind": "numeric", "min": 0, "max": 120}},
).fit(X)
validation = contract.validate(X)

evaluation = EvaluationReport(model, task="classification", random_state=0).run(
    X, y, cv="stratified", n_bootstrap=200,
)
```

`quality_report` summarizes missing values, duplicates, constant columns,
categorical frequencies, numeric outliers, and target quality. Use
`compare_schema` or `drift_report` for train/test checks. `AutoPreprocessor`
detects numeric and categorical columns, imputes missing values, scales numeric
features, one-hot encodes categorical features, and ignores unseen categories
by default.
`DataContract` adds explicit range, category, missing-rate, uniqueness, and
schema checks for training and inference boundaries. `TabularExperiment` can
accept it through `data_contract=` and includes the validation result in its
report.
`EvaluationReport` adds OOF predictions, fold scores, bootstrap confidence
intervals, learning curves, and convenient stratified, grouped, or time-series
CV selection.

`OnlineTabularExperiment` supports incremental `partial_fit`, bounded sliding
windows, prequential batch metrics, drift alerts, and delayed-label updates.

### Mathematical signal analysis

```python
from mastermlx.signal import (
    butterworth,
    ar_spectrum,
    arma_spectrum,
    cyclic_spectrum,
    cwt,
    coherence,
    envelope_spectrum,
    esprit,
    extract_ridge,
    frequency_response,
    group_delay,
    instantaneous_features,
    phase_response,
    pole_zero,
    prony,
    real_cepstrum,
    signal_cross_correlation,
    verify_filter,
    welch_psd,
)

freq, psd = welch_psd(x, sample_rate=1000, nperseg=256)
freq, coh = coherence(x, y, sample_rate=1000, nperseg=256)
b, a = butterworth(4, 100, sample_rate=1000, btype="lowpass")
freq, response = frequency_response(b, a, sample_rate=1000)
_, phase = phase_response(b, a, sample_rate=1000)
_, delay = group_delay(b, a, sample_rate=1000)
zeros, poles, gain = pole_zero(b, a)
report = verify_filter(
    b, a, sample_rate=1000,
    passband=(0, 50), stopband=(200, 500), stopband_db=20,
)

features = instantaneous_features(x, sample_rate=1000)
scales, frequencies, coefficients = cwt(x, scales=[4, 8, 16, 32], sample_rate=1000)
ridge = extract_ridge(np.abs(coefficients) ** 2, frequencies)

quefrency, cepstrum = real_cepstrum(x, sample_rate=1000)
env_freq, env_spec = envelope_spectrum(x, sample_rate=1000)
corr_lags, corr = signal_cross_correlation(x, y)
cyclic_freq, alpha, cyclic = cyclic_spectrum(x, [0, 20], sample_rate=1000)
ar_freq, ar_psd = ar_spectrum(x, order=8, sample_rate=1000)
arma_freq, arma_psd = arma_spectrum(x, ar_order=4, ma_order=2, sample_rate=1000)
prony_modes = prony(x, order=4, sample_rate=1000)
esprit_modes = esprit(x, n_components=2, sample_rate=1000)
```

These tools cover Welch power spectral density, cross-spectral coherence, and
frequency-domain analysis of discrete-time linear systems. IIR coefficients use
the standard increasing-delay convention, ``b[0] + b[1] z^-1 + ...`` and
``a[0] + a[1] z^-1 + ...``. The system tools provide magnitude and phase
responses, numerical group delay, pole-zero locations, unit-circle stability
checks, causal filtering, zero-phase filtering, and Butterworth lowpass,
highpass, bandpass, and bandstop designs. ``verify_filter`` returns the full
response arrays together with passband/stopband measurements and stability
metadata.

The time-frequency tools provide FFT-based Hilbert analytic signals,
instantaneous amplitude/phase/frequency, Morlet and Mexican-hat continuous
wavelet transforms, wavelet power maps, and dynamic-programming ridge
extraction. CWT scales are measured in samples; the returned frequencies are
pseudo-frequencies suitable for comparing dominant components over time.

### Neural-network module and acceleration layer

Neural-network layers share a small PyTorch-inspired module interface:

```python
from mastermlx.neural_net import Dense, ReLU, Sequential

model = Sequential([Dense(8, 16), ReLU(), Dense(16, 3)])
state = model.state_dict()
model.save("model.npz")
model.load("model.npz")
model.eval()
```

`Module` exposes `parameters`, `named_parameters`, `state_dict`,
`load_state_dict`, `train`, `eval`, `save`, and `load` while preserving the
existing NumPy-array layer implementation. `Conv1D` uses a vectorized NumPy
window path and has an optional Cython packing kernel. `SimpleRNN`, `LSTM`, and
`GRU` have optional Cython forward kernels with NumPy fallbacks; the fallbacks
remain available when compiled extensions are not installed. Signal IIR
filtering and time-frequency ridge extraction use the same backend-aware
pattern. `backend_report()` exposes the active backend and compiled capability
status for diagnostics. Training supports gradient clipping and accumulation,
configurable accuracy/precision/recall/F1/AUC or regression metrics, multilabel
classification, multi-output regression, and `ModelCheckpoint` full-object
checkpoints. `save_checkpoint()` preserves optimizer and scheduler state for
trusted local resume workflows; `save()` remains the portable parameter-only
`.npz` format.

Advanced spectral tools cover real/complex cepstrum analysis, envelope
demodulation and envelope spectra, cyclic spectral density, lag-domain
correlation and peak extraction, AR/ARMA spectra, and Prony/ESPRIT modal
frequency estimation. Parametric estimators return model coefficients or
modal dictionaries so their stability, damping, and frequency estimates can
be inspected directly.

Condition-monitoring helpers provide a business-level path from raw sensor
data to interpretable features:

```python
from mastermlx.signal import (
    SignalHealthMonitor,
    VibrationFeatureTransformer,
    signal_quality_report,
    vibration_features,
    windowed_vibration_features,
)

quality = signal_quality_report(signal, sample_rate=1000, saturation_level=2.0)
features = vibration_features(signal, sample_rate=1000, n_fft=2048)
transformer = VibrationFeatureTransformer(sample_rate=1000, n_fft=256)
feature_matrix = transformer.fit_transform(batch_of_signals)

monitor = SignalHealthMonitor(
    sample_rate=1000,
    feature_limits={"rms": (0.2, 1.2), "dominant_frequency": (40, 60)},
)
assessment = monitor.assess(signal)
windows = windowed_vibration_features(
    signal, sample_rate=1000, window_length=256, hop_length=128,
)
```

The quality report tracks finite-sample ratio, missing/infinite samples, RMS,
crest factor, saturation, duration, and optional SNR/PSNR. Vibration features
combine time-domain statistics, dominant frequency, spectral centroid and
bandwidth, zero-crossing rate, and normalized band energies. The transformer
is compatible with `SignalFeatureTransformer` and `SignalExperiment`.
`SignalHealthMonitor` adds explicit feature limits, health scores, status values
(`healthy`, `warning`, or `critical`), and machine-readable violations. Use
`windowed_vibration_features` to turn a long channel into timestamped feature
matrices for dashboards, alarm histories, or downstream models.

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
- `forward_kinematics_batch`
- `inverse_kinematics`
- `geometric_jacobian`
- `cubic_time_scaling`
- `quintic_time_scaling`
- `joint_trajectory`
- `plot_chain`
- `plan_joint_path`
- `smooth_joint_path`
- `plan_joint_trajectory`
- `RobotWorkcell` for joint-limited TCP and continuous Cartesian task planning, collision-aware inspection coverage, clearance-aware paths, velocity/acceleration/jerk-constrained retiming, singularity diagnostics, virtual tracking, and safety exports
- `RobotModel` exposes both concise (`fk`, `ik`, `jacobian`) and canonical long-form names, plus compiled batch FK/Jacobian evaluation
- `RobotModel.positions_batch` provides batched end-effector workspace positions
- `RobotModel.end_effector_velocity` and `RobotModel.end_effector_velocity_batch` provide velocity-level task-space mappings
- `RobotModel.differential_ik` provides damped least-squares velocity-level task-space control
- `RobotModel.ik_batch` and `inverse_kinematics_batch` solve target sequences with optional warm-starting and joint limits
- `RobotModel.joint_limits` enables limit-aware FK/Jacobian validation, clipping, and projected IK; URDF `<limit>` entries are imported when available
- `RobotModel.default_joint_values` selects zero when valid and otherwise the limit midpoint for safe default evaluation
- `RobotWorkcell` automatically inherits model joint limits unless explicit workcell limits are supplied
- `RobotModel.kinematic_metrics` for Jacobian rank, condition number, and manipulability diagnostics
- `RobotResult` and `JointTrajectory` for mapping-compatible, attribute-accessible workflow results
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
