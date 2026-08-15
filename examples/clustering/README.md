# Clustering Tutorial

[`kmeans_demo.py`](kmeans_demo.py) fits K-means with several cluster counts,
compares inertia and silhouette score, and saves a visual diagnostic.

## Run

```bash
python -m pip install "mastermlx[viz]==0.1.15"
python examples/clustering/kmeans_demo.py
```

The output is saved as `examples/outputs/kmeans_demo.png`.

## Interface

```python
from mastermlx.clustering import KMeans
from mastermlx.math_tools import silhouette

model = KMeans(n_clusters=3, random_state=0).fit(X)
labels = model.predict(X)
compactness = model.inertia_
separation = silhouette(X, labels)
```

Clustering is usually unsupervised, so `fit` receives `X` without `y`.
`labels_` describes the fitted rows and `predict(X_new)` assigns supported new
rows. Always set `random_state` when comparing stochastic clusterers.

See the [`clustering API index`](../API_REFERENCE.md#clustering-and-representation-learning)
for density, mixture, hierarchical, spectral, and representation-learning
alternatives.
