# Mathematical Tools Tutorial

[`math_tools_demo.py`](math_tools_demo.py) demonstrates the NumPy-first
statistics, similarity, clustering-quality, calibration, and time-series
helpers exported by `mastermlx.math_tools`.

```bash
python -m pip install "mastermlx[viz]==0.1.15"
MPLBACKEND=Agg python examples/tools/math_tools_demo.py
```

The plot is written to `examples/outputs/math_tools_demo.png`. Mathematical
helpers are functions rather than fitted estimators in most cases:

```python
from mastermlx.math_tools import entropy, rbf_kernel, silhouette

information = entropy(probabilities)
kernel_matrix = rbf_kernel(X, X, gamma=0.5)
cluster_quality = silhouette(X, labels)
```

Use `mastermlx.utils` for core estimator utilities, validation, and common
prediction metrics. See the
[`metrics and tools API index`](../API_REFERENCE.md#metrics-and-mathematical-tools).
