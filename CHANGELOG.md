# Changelog

## Unreleased

- Added a roadmap for future Cython / C++ optimization work and benchmark batches.
- Added Cython acceleration for time-series hot paths such as rolling mean, autocorrelation, exponential smoothing, and CUSUM detection.

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
