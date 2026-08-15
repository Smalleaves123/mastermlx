# Regression Tutorial

[`regression_pipeline.py`](regression_pipeline.py) is an end-to-end 0.1.15
workflow: generate data, split it, scale features, fit ridge regression, and
evaluate held-out predictions.

## Run

```bash
python -m pip install "mastermlx==0.1.15"
python examples/regression/regression_pipeline.py
```

## Interface

```python
from mastermlx.linear_models import RidgeRegression
from mastermlx.preprocessing import Pipeline, StandardScaler

model = Pipeline([
    ("scale", StandardScaler()),
    ("regressor", RidgeRegression(alpha=1.0)),
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
r2 = model.score(X_test, y_test)
```

Regressor `score` is R². Use `mean_absolute_error` or
`root_mean_squared_error` from `mastermlx.utils` when the error scale is more
useful. Pipeline parameters use `step__parameter`:

```python
model.set_params(regressor__alpha=0.5)
```

See the [`regression API index`](../API_REFERENCE.md#regression) for linear,
tree, neighbor, SVM, probabilistic, and neural regressors.
