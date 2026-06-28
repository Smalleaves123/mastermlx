"""
MLP on two interleaving spirals — loss curve + decision boundary.

Expected output:
    MLP spiral acc: ~0.60 (improves with more iters/layers)
"""
import numpy as np
import matplotlib.pyplot as plt
import mastermlx as mlx
from mastermlx.viz import plot_loss


def make_spirals(n=200, noise=0.1):
    t = np.linspace(0, 4 * np.pi, n)
    r = t / (4 * np.pi)
    X1 = np.column_stack([r * np.cos(t), r * np.sin(t)]) + noise * np.random.randn(n, 2)
    X2 = np.column_stack([r * np.cos(t + np.pi), r * np.sin(t + np.pi)]) + noise * np.random.randn(n, 2)
    return np.vstack([X1, X2]), np.array([0] * n + [1] * n)


X, y = make_spirals(300)
mlp = mlx.MLPClassifier(hidden_layer_sizes=(16, 8), lr=0.05, n_iter=500, random_state=0)
mlp.fit(X, y)
print(f"MLP spiral acc: {mlp.score(X, y):.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
plot_loss(mlp.loss_, ax=ax1, title="Training Loss")

# Decision boundary
xx = np.linspace(-1.5, 1.5, 80)
grid = np.column_stack([np.tile(xx, len(xx)), np.repeat(xx, len(xx))])
zz = mlp.predict(grid).reshape(len(xx), len(xx))
ax2.contourf(xx, xx, zz, alpha=0.3, cmap="RdBu")
ax2.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", s=15, edgecolor="k", linewidth=0.3)
ax2.set_title(f"Decision Boundary (acc={mlp.score(X, y):.3f})")
ax2.set_xlabel("x1"); ax2.set_ylabel("x2")

fig.tight_layout()
fig.savefig("examples/outputs/mlp_spirals.png", dpi=120, bbox_inches="tight")
plt.close()
print("→ Saved examples/outputs/mlp_spirals.png")
