"""
Benchmark the tabular workflow entry point.

This script keeps the benchmark small and deterministic so it can be used as a
development smoke test and as a product demo for the higher-level API.
"""
from __future__ import annotations

import time

import numpy as np

from mastermlx import LinearRegression, LogisticRegression, StandardScaler
from mastermlx.tabular import TabularExperiment, compare_tabular_models


def bench(fn, n_runs=3):
    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), result


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def make_classification_data(seed=42, n_samples=1200, n_features=12):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    logits = 1.6 * X[:, 0] - 1.1 * X[:, 1] + 0.7 * X[:, 2] + 0.5 * rng.normal(size=n_samples)
    y = (logits > 0).astype(int)
    return X, y


def make_regression_data(seed=24, n_samples=1600, n_features=10):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = 2.5 * X[:, 0] - 1.7 * X[:, 1] + 0.8 * X[:, 2] + 0.2 * rng.normal(size=n_samples)
    return X, y


def benchmark_classification():
    X, y = make_classification_data()

    direct = TabularExperiment(
        model=LogisticRegression(n_iter=400, lr=0.05, random_state=0),
        preprocessing=StandardScaler(),
        search=None,
        task="classification",
    )
    direct_time, _ = bench(lambda: direct.fit(X, y), n_runs=3)
    direct_score = direct.score(X, y)

    tuned = TabularExperiment(
        model=LogisticRegression(n_iter=400, lr=0.05, random_state=0),
        preprocessing=StandardScaler(),
        search="grid",
        param_grid={"model__lr": [0.01, 0.05, 0.1]},
        cv=3,
        task="classification",
    )
    tuned_time, _ = bench(lambda: tuned.fit(X, y), n_runs=2)
    tuned_score = tuned.score(X, y)

    print(f"  direct fit       {direct_time:8.4f}s  score={direct_score:6.3f}")
    print(f"  grid search      {tuned_time:8.4f}s  score={tuned_score:6.3f}")
    print(f"  best params      {tuned.best_params_}")
    return direct, tuned


def benchmark_regression():
    X, y = make_regression_data()
    result = compare_tabular_models(
        [
            ("linear_intercept", LinearRegression(fit_intercept=True)),
            ("linear_no_intercept", LinearRegression(fit_intercept=False)),
        ],
        X,
        y,
        preprocessing=StandardScaler(),
        task="regression",
    )
    print(f"  leaderboard      {result['leaderboard']}")
    print(f"  best model       {result['best_name']}  score={result['best_score']:.4f}")
    return result


def main():
    section("Tabular Classification Workflow")
    benchmark_classification()

    section("Tabular Regression Workflow")
    benchmark_regression()

    section("Summary")
    print("  This benchmark covers preprocessing, training, search, and comparison.")
    print("  It is designed as a lightweight product demo rather than a saturation test.")


if __name__ == "__main__":
    main()
