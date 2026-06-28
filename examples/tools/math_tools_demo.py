"""
Showcase math_tools: stats, distributions, similarity, noise.

Expected output:
    entropy(p)    = 1.0297
    KL(p||q)      = 0.1168
    mann-whitney  U=~3500 p=~0.0004
    pearson_r     = 1.0000
    Normal.cdf(0) = 0.5000
    Poisson.pmf(3)= 0.1404
    gauss:         [0.977 1.125 0.936 1.072 0.991]
    salt_pepper:   [ 1. -9.  9.  9.  1.]
    SMOTE: 60 → 100 (class 1: 50)
"""
import numpy as np
import matplotlib.pyplot as plt
from mastermlx import (
    entropy, kl_divergence, mann_whitney, pearson_r, PCA,
    Normal, Poisson, Exponential, Chi2, StudentT,
    gauss, salt_pepper, smote, mixup,
)

rng = np.random.default_rng(42)

# --- Print section ---
p = np.array([0.2, 0.3, 0.5])
q = np.array([0.4, 0.3, 0.3])
print(f"entropy(p)    = {entropy(p):.4f}")
print(f"KL(p||q)      = {kl_divergence(p, q):.4f}")

x, y = rng.normal(0, 1, 100), rng.normal(0.5, 1, 100)
u, p_val = mann_whitney(x, y)
print(f"mann-whitney  U={u:.2f} p={p_val:.4f}")
print(f"pearson_r     = {pearson_r(np.arange(50), 2*np.arange(50)+1):.4f}")

n = Normal(0, 1)
print(f"Normal.cdf(0) = {n.cdf(0):.4f}")
pois = Poisson(lam=5)
print(f"Poisson.pmf(3)= {pois.pmf(3):.4f}")

data = np.ones(5)
print(f"gauss:         {gauss(data, 0.1, random_state=0).round(3)}")
print(f"salt_pepper:   {salt_pepper(data, 0.4, salt_val=9, pepper_val=-9, random_state=0)}")

X_imb = np.vstack([rng.normal(0, 1, (50, 3)), rng.normal(3, 1, (10, 3))])
y_imb = np.array([0] * 50 + [1] * 10)
X_bal, y_bal = smote(X_imb, y_imb, k=5, random_state=0)
print(f"SMOTE: {X_imb.shape[0]} → {X_bal.shape[0]} (class 1: {np.sum(y_bal == 1)})")

# --- Plot section ---
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# 1. Distributions
xs = np.linspace(-4, 4, 200)
for dist, label in [(Normal(0, 1), "N(0,1)"), (StudentT(4), "t(4)")]:
    axes[0, 0].plot(xs, dist.pdf(xs), label=label)
axes[0, 0].set_title("PDF: Normal vs Student-t")
axes[0, 0].legend()

# 2. Discrete distributions
ks = np.arange(0, 15)
for dist, label in [(Poisson(3), "Poisson(3)"), (Poisson(7), "Poisson(7)")]:
    axes[0, 1].bar(ks - 0.15 if "3" in label else ks + 0.15, dist.pmf(ks), width=0.3, label=label, alpha=0.7)
axes[0, 1].set_title("PMF: Poisson")
axes[0, 1].legend()

# 3. Correlation
xa = np.arange(30) + rng.normal(0, 2, 30)
ya = 2 * xa + rng.normal(0, 2, 30)
axes[0, 2].scatter(xa, ya, alpha=0.6, s=20)
axes[0, 2].set_title(f"pearson_r={pearson_r(xa, ya):.3f}")

# 4. Noise
orig = rng.normal(0, 1, 50)
axes[1, 0].plot(orig, "o-", ms=4, label="original", alpha=0.6)
axes[1, 0].plot(gauss(orig, 0.5, random_state=0), "s-", ms=3, label="+ gauss(0.5)", alpha=0.6)
axes[1, 0].set_title("Gaussian Noise")
axes[1, 0].legend(fontsize=8)

# 5. MixUp
X_mix = rng.normal(0, 1, (60, 2))
Xa, yp, w = mixup(X_mix, np.arange(60), alpha=0.5, random_state=0)
axes[1, 1].scatter(X_mix[:, 0], X_mix[:, 1], c="gray", s=15, alpha=0.4, label="original")
axes[1, 1].scatter(Xa[:, 0], Xa[:, 1], c="red", s=10, alpha=0.5, label="mixed")
axes[1, 1].set_title("MixUp Augmentation")
axes[1, 1].legend(fontsize=8)

# 6. SMOTE
X_pca = np.random.default_rng(1).normal(0, 1, (60, 20))
y_pca = np.array([0]*50 + [1]*10)
X_sm, _ = smote(X_pca, y_pca, k=5, random_state=0)
pca_model = PCA(n_components=2).fit(np.vstack([X_pca, X_sm]))
Xp, Xs = pca_model.transform(X_pca[y_pca==0]), pca_model.transform(X_pca[y_pca==1])
Xsyn = pca_model.transform(X_sm[len(X_pca):])
axes[1, 2].scatter(Xp[:,0], Xp[:,1], c="steelblue", s=12, alpha=0.5, label="majority")
axes[1, 2].scatter(Xs[:,0], Xs[:,1], c="darkorange", s=15, label="minority")
axes[1, 2].scatter(Xsyn[:,0], Xsyn[:,1], c="red", s=10, alpha=0.7, label="synthetic")
axes[1, 2].set_title(f"SMOTE (PCA view)")
axes[1, 2].legend(fontsize=7)

fig.suptitle("math_tools Showcase", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig("examples/outputs/math_tools_demo.png", dpi=120, bbox_inches="tight")
plt.close()
print("\n→ Saved examples/outputs/math_tools_demo.png")
