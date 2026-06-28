from __future__ import annotations

import numpy as np


def resolve_gamma(gamma, n_features):
    if gamma is None or gamma == "scale":
        return 1.0 / max(int(n_features), 1)
    return float(gamma)


def linear_kernel(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    return X @ Y.T


def cosine_kernel(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True).T
    return (X @ Y.T) / (X_norm * Y_norm + 1e-12)


def poly_kernel(X, Y, gamma, coef0, degree):
    return (gamma * linear_kernel(X, Y) + coef0) ** degree


def rbf_kernel(X, Y, gamma):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    x2 = np.sum(X ** 2, axis=1)[:, None]
    y2 = np.sum(Y ** 2, axis=1)[None, :]
    d2 = np.maximum(x2 + y2 - 2.0 * (X @ Y.T), 0.0)
    return np.exp(-gamma * d2)


def laplacian_kernel(X, Y, gamma):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    l1 = np.sum(np.abs(X[:, None, :] - Y[None, :, :]), axis=2)
    return np.exp(-gamma * l1)


def sigmoid_kernel(X, Y, gamma, coef0):
    return np.tanh(gamma * linear_kernel(X, Y) + coef0)


def chi2_kernel(X, Y, gamma):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if np.any(X < 0) or np.any(Y < 0):
        raise ValueError("chi2_kernel expects non-negative inputs")
    num = (X[:, None, :] - Y[None, :, :]) ** 2
    den = X[:, None, :] + Y[None, :, :] + 1e-12
    chi2 = 0.5 * np.sum(num / den, axis=2)
    return np.exp(-gamma * chi2)


def additive_chi2_kernel(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if np.any(X < 0) or np.any(Y < 0):
        raise ValueError("additive_chi2_kernel expects non-negative inputs")
    num = 2.0 * X[:, None, :] * Y[None, :, :]
    den = X[:, None, :] + Y[None, :, :] + 1e-12
    return np.sum(num / den, axis=2)


def hellinger_kernel(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if np.any(X < 0) or np.any(Y < 0):
        raise ValueError("hellinger_kernel expects non-negative inputs")
    return np.sum(np.sqrt(X)[:, None, :] * np.sqrt(Y)[None, :, :], axis=2)


def pairwise_kernel(X, Y, kernel="rbf", gamma=None, coef0=0.0, degree=3):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    kernel = kernel.lower()
    gamma = resolve_gamma(gamma, X.shape[1])
    if kernel == "linear":
        return linear_kernel(X, Y)
    if kernel == "cosine":
        return cosine_kernel(X, Y)
    if kernel == "poly":
        return poly_kernel(X, Y, gamma, coef0, degree)
    if kernel == "rbf":
        return rbf_kernel(X, Y, gamma)
    if kernel == "laplacian":
        return laplacian_kernel(X, Y, gamma)
    if kernel == "sigmoid":
        return sigmoid_kernel(X, Y, gamma, coef0)
    if kernel == "chi2":
        return chi2_kernel(X, Y, gamma)
    if kernel == "additive_chi2":
        return additive_chi2_kernel(X, Y)
    if kernel == "hellinger":
        return hellinger_kernel(X, Y)
    raise ValueError(
        "kernel must be one of: linear, cosine, poly, rbf, laplacian, sigmoid, chi2, additive_chi2, hellinger"
    )
