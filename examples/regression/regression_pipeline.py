"""End-to-end regression with splitting, preprocessing, and evaluation.

Run from the repository root:

    python examples/regression/regression_pipeline.py
"""

import numpy as np

from mastermlx.data import train_test_split
from mastermlx.linear_models import RidgeRegression
from mastermlx.preprocessing import Pipeline, StandardScaler
from mastermlx.utils import mean_absolute_error, root_mean_squared_error


def make_dataset(random_state=42):
    """Create a reproducible nonlinear regression dataset."""

    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(400, 4))
    y = 3.0 * X[:, 0] - 1.5 * X[:, 1] + 0.8 * X[:, 2] ** 2
    y += rng.normal(scale=0.35, size=X.shape[0])
    return X, y


def main():
    X, y = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=7,
    )

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("regressor", RidgeRegression(alpha=1.0)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"train samples: {X_train.shape[0]}")
    print(f"test samples:  {X_test.shape[0]}")
    print(f"R2:   {model.score(X_test, y_test):.3f}")
    print(f"MAE:  {mean_absolute_error(y_test, predictions):.3f}")
    print(f"RMSE: {root_mean_squared_error(y_test, predictions):.3f}")

    # A single sample is still represented as a 2D feature matrix. The output
    # keeps its sample axis and therefore has shape (1,).
    one_prediction = model.predict(X_test[:1])
    print("single prediction shape:", one_prediction.shape)

    # Nested parameters use the same ``step__parameter`` convention as the
    # rest of the estimator API.
    print("configured alpha:", model.get_params()["regressor__alpha"])


if __name__ == "__main__":
    main()
