"""
TF-IDF text classification — confusion matrix + top features bar chart.

Expected output:
    TF-IDF + LR acc: ~1.00
    Features: ['the', 'cat', 'dog', 'fish', 'the cat', 'the dog', ...]
"""
import numpy as np
import matplotlib.pyplot as plt
import mastermlx as mlx
from mastermlx.viz import plot_cm

docs = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the cat ate fish",
    "the dog ate meat",
    "cat fish food",
    "dog meat bone",
    "the cat likes fish",
    "the dog likes meat",
    "fish swim in water",
    "dogs run in parks",
]
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 0 → cat, 1 → dog

vec = mlx.TfidfVectorizer(ngram_range=(1, 2), max_features=50)
X = vec.fit_transform(docs)
clf = mlx.LogisticRegression(lr=0.1, n_iter=500).fit(X, labels)

print(f"TF-IDF + LR acc: {clf.score(X, labels):.2f}")
print("Features:", vec.feature_names_[:10])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Confusion matrix
plot_cm(labels, clf.predict(X), title="TF-IDF + LogisticRegression", ax=ax1)

# Top features by absolute coefficient weight
coef = np.abs(clf.coef_)
top_k = min(15, len(coef))
top_idx = np.argsort(coef)[-top_k:]
ax2.barh(range(top_k), coef[top_idx], color="#4C72B0")
ax2.set_yticks(range(top_k))
ax2.set_yticklabels([vec.feature_names_[i] for i in top_idx])
ax2.set_title("Top Features (by |weight|)")
ax2.set_xlabel("|coefficient|")

fig.tight_layout()
fig.savefig("examples/outputs/text_classify.png", dpi=120, bbox_inches="tight")
plt.close()
print("→ Saved examples/outputs/text_classify.png")
