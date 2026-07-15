# Changelog

## 0.1.13

- Optimized the pure NumPy squared-Euclidean distance fallback with a BLAS-friendly matrix formulation.
- Removed the large three-dimensional broadcast buffer from the NumPy distance path, reducing memory usage and improving fallback performance.
- Made default time-series cross-validation adapt to short series while validating the minimum training-fold size.
- Added regression coverage for the optimized distance fallback and short-series model comparison.

## Unreleased

- Added cepstral analysis and cepstrum-peak extraction, Hilbert envelope
  demodulation, envelope spectra, and cyclic spectral-density estimation.
- Added signal autocorrelation/cross-correlation with lag-aware peak analysis.
- Added Yule-Walker AR, Hannan-Rissanen ARMA, Prony, and ESPRIT estimators with
  spectra, modal frequencies, damping, and amplitude outputs.
- Added FFT-based Hilbert analytic signals with instantaneous amplitude, phase,
  and frequency extraction.
- Added NumPy continuous wavelet transforms with Morlet and Mexican-hat
  wavelets, scale generation, wavelet power maps, and time-frequency ridge
  extraction.
- Added IIR system analysis for magnitude, phase, and group-delay responses,
  pole-zero inspection, stability checks, causal filtering, and zero-phase
  filtering.
- Added NumPy-only Butterworth lowpass, highpass, bandpass, and bandstop
  design through bilinear transformation, with normalized passband gain.
- Added filter verification reports with response arrays, pole-radius metrics,
  and optional passband-ripple and stopband-attenuation checks.
- Added mathematical signal-analysis tools for Welch PSD, periodograms, cross spectra, coherence, discrete-system frequency response, and group delay.
- Split tabular workflows and data quality/schema/drift checks into focused implementation modules while preserving existing public imports.
- Added an optional C++ histogram-tree kernel for `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor`, with numerical parity and a NumPy fallback.
- Made backend loaders honor `set_backend("numpy")` and `set_backend("cython")` instead of using compiled modules unconditionally.
- Corrected acceleration benchmarks to switch backends through the public API, report missing Cython extensions accurately, and keep development runs practical.
- Added Cython acceleration for `confusion_matrix` counting in `mastermlx.utils.metrics`.
- Added Cython acceleration for particle-filter systematic resampling and weight normalization.
- Added Cython acceleration for linear, cosine, polynomial, and sigmoid kernels.
- Added Cython acceleration for scalar distance helpers in `mastermlx.utils.distance`.
- Added Cython acceleration for linear Kalman predict/update and shared EKF matrix updates.
- Added Cython acceleration for cosine, hamming, jaccard, and Mahalanobis pairwise distances.
- Routed RBF kernel evaluation through the C++ backend in `mastermlx.utils.kernels`.
- Added Cython acceleration for discrete LQR recursion and batch joint-trajectory sampling.
- Added roadmap guidance for the next Cython / C++ optimization batches.
- Added Cython acceleration for time-series hot paths such as rolling mean, autocorrelation, exponential smoothing, and CUSUM detection.

## 0.1.12

- Continued the compiled acceleration pass into particle-filter utilities.

## 0.1.11

- Continued the compiled acceleration pass into the remaining high-frequency pairwise kernels.

## 0.1.10

- Continued the compiled acceleration pass into one-to-one distance helpers.

## 0.1.9

- Continued the compiled acceleration pass into state-estimation matrix updates.

## 0.1.8

- Continued the compiled acceleration pass into distance metrics that previously used 3D NumPy broadcasts.

## 0.1.7

- Added compiled hot-path helpers for control, robotics, and time-series routines.
- Continued the Cython optimization pass into LQR and trajectory sampling.

## 0.1.6

- Added Cython acceleration for robotics hot paths in `mastermlx.control` and `mastermlx.robotics`.
- Moved iLQR rollout, finite-difference Jacobians, and trajectory cost accumulation off the Python slow path when Cython is available.
- Added Cython-accelerated forward kinematics, geometric Jacobian, and inverse kinematics packing for serial manipulators.

## 0.1.5

- Added a new `mastermlx.control` package with PID, LQR, finite-horizon MPC, and iLQR optimization control.
- Continued the robotics stack with optimization-based control foundations.

## 0.1.4

- Added a new `mastermlx.estimation` package for Kalman, extended Kalman, and particle filtering.
- Continued the robotics stack with state estimation foundations.

## 0.1.3

- Restored non-negative input validation for chi-squared and Hellinger kernels.
- Made Hellinger kernel numerically stable for large values.
- Restored positive-exponent validation for Minkowski pairwise distances.

- Added a new `mastermlx.robotics` foundation package for transforms, kinematics, Jacobians, trajectories, and visualization.

## 0.1.1

- Tightened packaging metadata for PyPI publication.
- Added and validated compiled acceleration for convolution and max pooling.
- Improved KNN and DBSCAN hot paths.
- Refreshed README content for end users and technical readers.

## 0.1.0

Initial public release of `mastermlx`.

- NumPy-first machine learning library
- Top-level API with classic ML algorithms
- Math tools for metrics, distance functions, kernels, statistics, and time series
- Optional C++ and Cython acceleration
