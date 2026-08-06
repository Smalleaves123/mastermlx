"""Benchmark mastermlx against SciPy and scikit-learn reference paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import warnings

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.decomposition import NMF as SklearnNMF
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.metrics import accuracy_score, adjusted_rand_score, mean_squared_error

from mastermlx import (
    KMeans,
    LinearRegression,
    LogisticRegression,
    NMF,
    PCA,
    RidgeRegression,
    get_backend,
    set_backend,
)
from mastermlx.accel import pairwise_squared_euclidean

warnings.filterwarnings("ignore")


def _measure(function, repeats):
    function()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        values.append(time.perf_counter() - start)
    return float(np.median(values))


def _fit_score(
    name,
    ours_factory,
    reference_factory,
    X_train,
    y_train,
    X_test,
    y_test,
    score,
    quality_label,
):
    ours_time = _measure(lambda: ours_factory().fit(X_train, y_train), repeats=3)
    reference_time = _measure(lambda: reference_factory().fit(X_train, y_train), repeats=3)
    ours_model = ours_factory().fit(X_train, y_train)
    reference_model = reference_factory().fit(X_train, y_train)
    ours_score = score(ours_model, X_test, y_test)
    reference_score = score(reference_model, X_test, y_test)
    result = {
        "name": name,
        "mastermlx_seconds": ours_time,
        "reference_seconds": reference_time,
        "time_ratio": ours_time / reference_time,
        "mastermlx_quality": ours_score,
        "reference_quality": reference_score,
        "quality_label": quality_label,
    }
    print(
        f"{name:24s} mastermlx={ours_time:8.5f}s  sklearn={reference_time:8.5f}s  "
        f"time={ours_time / reference_time:6.2f}x  "
        f"{quality_label}={ours_score:.6f}/{reference_score:.6f}"
    )
    return result


def _linear_quality(model, X, y):
    return mean_squared_error(y, model.predict(X))


def _classification_quality(model, X, y):
    return accuracy_score(y, model.predict(X))


def _cluster_quality(model, X, labels):
    return adjusted_rand_score(labels, model.predict(X))


def _pca_quality(model, X, _unused):
    transformed = model.transform(X)
    reconstructed = model.inverse_transform(transformed)
    return mean_squared_error(X, reconstructed)


def _nmf_quality(model, X, _unused):
    reconstructed = model.transform(X) @ model.components_
    return mean_squared_error(X, reconstructed)


def _scipy_lstsq(X, y):
    augmented = np.column_stack([np.ones(X.shape[0]), X])
    linalg.lstsq(augmented, y, lapack_driver="gelsd")


def _scipy_ridge(X, y, alpha):
    centered_X = X - np.mean(X, axis=0)
    centered_y = y - np.mean(y)
    gram = centered_X.T @ centered_X
    linalg.solve(gram + alpha * np.eye(X.shape[1]), centered_X.T @ centered_y, assume_a="pos")


def _scipy_pca(X, n_components):
    centered = X - np.mean(X, axis=0)
    _, _, vectors = linalg.svd(centered, full_matrices=False, lapack_driver="gesdd")
    return centered @ vectors[:n_components].T


def _print_scipy_primitives(rng):
    print("\nSciPy numerical primitives")
    X_reg, y_reg = make_regression(
        n_samples=4000, n_features=24, n_informative=16, noise=1.0, random_state=42
    )
    X_pca = rng.normal(size=(2500, 60))
    X_distance = rng.normal(size=(1800, 20))
    Y_distance = rng.normal(size=(900, 20))

    comparisons = (
        (
            "lstsq",
            lambda: LinearRegression().fit(X_reg, y_reg),
            lambda: _scipy_lstsq(X_reg, y_reg),
        ),
        (
            "ridge solve",
            lambda: RidgeRegression(alpha=1.0).fit(X_reg, y_reg),
            lambda: _scipy_ridge(X_reg, y_reg, 1.0),
        ),
        (
            "PCA SVD",
            lambda: PCA(8).fit_transform(X_pca),
            lambda: _scipy_pca(X_pca, 8),
        ),
        (
            "pairwise squared distance",
            lambda: pairwise_squared_euclidean(X_distance, Y_distance),
            lambda: cdist(X_distance, Y_distance, metric="sqeuclidean"),
        ),
    )
    results = []
    for name, ours, scipy_function in comparisons:
        ours_time = _measure(ours, repeats=3)
        scipy_time = _measure(scipy_function, repeats=3)
        result = {
            "name": name,
            "mastermlx_seconds": ours_time,
            "reference_seconds": scipy_time,
            "time_ratio": ours_time / scipy_time,
            "reference": "scipy",
        }
        results.append(result)
        print(
            f"{name:24s} mastermlx={ours_time:8.5f}s  scipy={scipy_time:8.5f}s  "
            f"time={ours_time / scipy_time:6.2f}x"
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("auto", "numpy", "cython"), default="auto", help="mastermlx backend"
    )
    parser.add_argument(
        "--json-output", type=Path, help="write benchmark results to a JSON file"
    )
    args = parser.parse_args()
    set_backend(args.backend)
    rng = np.random.default_rng(42)
    print(f"mastermlx backend: {get_backend()}")
    results = []

    X, y = make_classification(
        n_samples=5000, n_features=20, n_informative=10, random_state=42
    )
    split = 4000
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print("\nscikit-learn estimators")
    results.append(_fit_score(
        "LogisticRegression",
        lambda: LogisticRegression(lr=0.1, n_iter=100, random_state=0),
        lambda: SklearnLogisticRegression(max_iter=100, random_state=0),
        X_train,
        y_train,
        X_test,
        y_test,
        _classification_quality,
        "accuracy",
    ))

    X_blobs, labels = make_blobs(n_samples=3000, n_features=10, centers=5, random_state=42)
    results.append(_fit_score(
        "KMeans",
        lambda: KMeans(5, n_init=1, random_state=0),
        lambda: SklearnKMeans(5, n_init=1, random_state=0),
        X_blobs,
        labels,
        X_blobs,
        labels,
        _cluster_quality,
        "ARI",
    ))

    X_reg, y_reg = make_regression(
        n_samples=3000, n_features=15, n_informative=10, noise=1.0, random_state=42
    )
    results.append(_fit_score(
        "LinearRegression",
        LinearRegression,
        SklearnLinearRegression,
        X_reg[:2400],
        y_reg[:2400],
        X_reg[2400:],
        y_reg[2400:],
        _linear_quality,
        "MSE",
    ))
    results.append(_fit_score(
        "RidgeRegression",
        lambda: RidgeRegression(alpha=1.0),
        lambda: SklearnRidge(alpha=1.0),
        X_reg[:2400],
        y_reg[:2400],
        X_reg[2400:],
        y_reg[2400:],
        _linear_quality,
        "MSE",
    ))

    X_pca = rng.normal(size=(2000, 50))
    results.append(_fit_score(
        "PCA",
        lambda: PCA(5),
        lambda: SklearnPCA(5),
        X_pca,
        None,
        X_pca,
        None,
        _pca_quality,
        "reconstruction_MSE",
    ))

    X_nmf = np.abs(rng.normal(size=(1200, 30)))
    results.append(_fit_score(
        "NMF",
        lambda: NMF(5, max_iter=100, random_state=0),
        lambda: SklearnNMF(5, max_iter=100, random_state=0),
        X_nmf,
        None,
        X_nmf,
        None,
        _nmf_quality,
        "reconstruction_MSE",
    ))

    results.extend(_print_scipy_primitives(rng))

    if args.json_output is not None:
        payload = {
            "backend": get_backend(),
            "results": results,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
