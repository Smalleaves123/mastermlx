"""
Compare classifiers + plot confusion matrices.

Expected output:
    LogisticRegression         acc: ~0.97
    SGDClassifier              acc: ~0.95
    RandomForest               acc: ~0.97
    SVC                        acc: ~0.97
"""
import numpy as np
import matplotlib.pyplot as plt
import mastermlx as mlx
from mastermlx.viz import plot_cm

rng = np.random.default_rng(42)
X = rng.normal(size=(500, 10))
y = np.where(X[:, 0] + X[:, 1] - 0.5 * X[:, 2] > 0, 1, 0)

models = {
    "LogisticRegression": mlx.LogisticRegression(lr=0.1, n_iter=200),
    "SGDClassifier":      mlx.SGDClassifier(loss="hinge", max_iter=100),
    "RandomForest":       mlx.RandomForestClassifier(n_estimators=50, max_depth=5),
    "SVC":                mlx.SVC(C=1.0, kernel="rbf", max_iter=500),
}

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for ax, (name, model) in zip(axes.flat, models.items()):
    model.fit(X, y)
    acc = model.score(X, y)
    print(f"{name:25s}  acc: {acc:.4f}")
    plot_cm(y, model.predict(X), title=f"{name} ({acc:.3f})", ax=ax)

fig.suptitle("Confusion Matrices — 4 Classifiers", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig("examples/outputs/compare_models.png", dpi=120, bbox_inches="tight")
plt.close()
print("\n→ Saved examples/outputs/compare_models.png")
