"""
KMeans clustering + scatter plot + silhouette.

Expected output:
    k=2  inertia=~1000  silhouette=~0.54
    k=3  inertia=~200   silhouette=~0.74  ← best
    k=4  inertia=~170   silhouette=~0.61
"""
import numpy as np
import matplotlib.pyplot as plt
import mastermlx as mlx
from mastermlx import silhouette
from mastermlx.viz import plot_scatter

rng = np.random.default_rng(42)
X = np.vstack([
    rng.normal(0, 0.6, (100, 2)),
    rng.normal(0, 0.6, (100, 2)) + [4, 0],
    rng.normal(0, 0.6, (100, 2)) + [2, 4],
])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for k, ax in zip([2, 3, 4], axes):
    km = mlx.KMeans(n_clusters=k, random_state=0).fit(X)
    s = silhouette(X, km.labels_)
    print(f"k={k}  inertia={km.inertia_:.2f}  silhouette={s:.3f}")
    plot_scatter(X, km.labels_, ax=ax, title=f"k={k}  sil={s:.3f}")

fig.suptitle("KMeans Clustering", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig("examples/outputs/kmeans_demo.png", dpi=120, bbox_inches="tight")
plt.close()
print("\n→ Saved examples/outputs/kmeans_demo.png")
