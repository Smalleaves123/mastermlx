# `mastermlx.math_tools`

Unified NumPy-first mathematical utilities for machine learning.

## What is here

- `distance`: Euclidean, Manhattan, Minkowski, cosine, Mahalanobis, Hamming, Jaccard
- `kernels`: linear, polynomial, RBF, Laplacian, sigmoid, chi-square, Hellinger
- `information`: entropy, cross-entropy, KL, JS, mutual information
- `metrics`: accuracy, balanced accuracy, MCC, Cohen kappa, hinge loss
- `calibration`: Brier score, ECE, MCE, reliability curve
- `attention`: scaled dot-product attention, multi-head attention, positional encoding
- `time_series`: AR model, differencing, rolling mean, band-limited DTW, CUSUM change points

## Import

```python
from mastermlx import math_tools as mt

dist = mt.cosine_distance([1, 0], [0, 1])
weights = mt.scaled_dot_product_attention(q, k, v)
```

The public API is NumPy-first and remains usable without compiled extensions;
optional Cython/C++ backends accelerate selected hot paths.

DTW accepts an optional non-negative `window`.  A narrow window reduces the
dynamic-programming memory footprint while keeping the NumPy and compiled
backends numerically consistent.  `accel.kdtree.knn_search` returns Euclidean
distances and automatically falls back to NumPy when the C++ extension is not
available.
