"""Synthetic dataset generators for quick experimentation."""

from __future__ import annotations

import numpy as np


def make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0,
               center_box=(-10.0, 10.0), random_state=None):
    """Generate isotropic Gaussian blobs for clustering / classification."""
    rng = np.random.default_rng(random_state)
    n_samples = int(n_samples)
    n_features = int(n_features)
    if isinstance(centers, int):
        c = int(centers)
        centroids = rng.uniform(center_box[0], center_box[1], size=(c, n_features))
    else:
        centroids = np.asarray(centers, dtype=float)
    if isinstance(cluster_std, (int, float)):
        cluster_std = [float(cluster_std)] * len(centroids)
    X, y = [], []
    for i, (ctr, std) in enumerate(zip(centroids, cluster_std)):
        n = n_samples // len(centroids) + (1 if i < n_samples % len(centroids) else 0)
        X.append(rng.normal(ctr, float(std), size=(n, n_features)))
        y.append(np.full(n, i))
    return np.vstack(X), np.concatenate(y).astype(int)


def make_moons(n_samples=100, noise=0.1, random_state=None):
    """Generate two interleaving half circles (moons)."""
    rng = np.random.default_rng(random_state)
    n_out = n_samples // 2
    n_in = n_samples - n_out

    outer = np.linspace(0, np.pi, n_out)
    inner = np.linspace(0, np.pi, n_in)

    X = np.vstack([
        np.column_stack([np.cos(outer), np.sin(outer)]),
        np.column_stack([1 - np.cos(inner), 1 - np.sin(inner) - 0.5]),
    ])
    y = np.array([0] * n_out + [1] * n_in)

    X += rng.normal(0, noise, size=X.shape)
    return X, y


def make_classification(n_samples=100, n_features=20, n_informative=2,
                         n_redundant=2, n_classes=2, flip_y=0.01,
                         random_state=None):
    """Generate a random n-class classification problem."""
    rng = np.random.default_rng(random_state)
    n_informative = min(n_informative, n_features)
    n_redundant = min(n_redundant, n_features - n_informative)

    # Informative features
    X_inf = rng.normal(size=(n_samples, n_informative))
    coef = rng.normal(size=(n_informative, n_classes))
    logits = X_inf @ coef
    y = np.argmax(logits + rng.normal(0, 0.1, size=logits.shape), axis=1).astype(int)

    # Redundant features (linear combinations of informative)
    X_red = X_inf[:, :n_redundant] @ rng.normal(size=(n_redundant, n_redundant))

    # Noise features
    n_noise = n_features - n_informative - n_redundant
    X_noise = rng.normal(size=(n_samples, max(0, n_noise)))

    X = np.column_stack([X_inf, X_red, X_noise]) if n_noise >= 0 else np.column_stack([X_inf, X_red])

    # Flip labels
    if flip_y > 0:
        mask = rng.random(n_samples) < flip_y
        y[mask] = rng.integers(0, n_classes, size=int(np.sum(mask)))
    return X, y


def make_regression(n_samples=100, n_features=10, n_informative=5,
                     noise=1.0, random_state=None):
    """Generate a random regression problem."""
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    coef = rng.normal(size=(n_features,))
    coef[n_informative:] = 0.0
    y = X @ coef + rng.normal(0, noise, size=n_samples)
    return X, y
