"""Probabilistic regression and classification interfaces.

Run from the repository root:

    python examples/probabilistic/probabilistic_models.py
"""

import numpy as np

from mastermlx.probabilistic import DiscriminantLDA, GaussianProcessRegressor


def regression_demo():
    """Fit a Gaussian process and inspect predictive uncertainty."""

    X_train = np.linspace(-3.0, 3.0, 18)[:, None]
    y_train = np.sin(X_train[:, 0])
    X_query = np.array([[-2.5], [0.0], [2.5], [5.0]])

    model = GaussianProcessRegressor(length_scale=1.0, alpha=1e-5).fit(
        X_train,
        y_train,
    )
    mean, std = model.predict(X_query, return_std=True)

    print("Gaussian process predictions:")
    for x, prediction, uncertainty in zip(X_query[:, 0], mean, std):
        print(f"  x={x:4.1f}  mean={prediction: .3f}  std={uncertainty:.3f}")
    print("posterior summary:", model.posterior_summary())


def classification_demo():
    """Use the explicit discriminant-analysis alias for classification."""

    rng = np.random.default_rng(12)
    X_negative = rng.normal(loc=(-1.5, -1.0), scale=0.45, size=(80, 2))
    X_positive = rng.normal(loc=(1.2, 1.5), scale=0.45, size=(80, 2))
    X = np.vstack([X_negative, X_positive])
    y = np.array(["negative"] * 80 + ["positive"] * 80)

    classifier = DiscriminantLDA().fit(X, y)
    predictions = classifier.predict([[0.9, 1.1]])
    print("DiscriminantLDA accuracy:", f"{classifier.score(X, y):.3f}")
    print("single prediction:", predictions, "shape=", predictions.shape)


def main():
    regression_demo()
    print()
    classification_demo()


if __name__ == "__main__":
    main()
