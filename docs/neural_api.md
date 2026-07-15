# Neural API and persistence

## Prediction shapes

Neural estimators keep the sample axis for every prediction call:

- classification: `predict(X)` returns `(n_samples,)`
- classification probabilities: `predict_proba(X)` returns `(n_samples, n_classes)`
- multilabel: `predict(X)` returns `(n_samples, n_labels)`
- single-output regression: `predict(X)` returns `(n_samples,)`
- multi-output regression: `predict(X)` returns `(n_samples, n_outputs)`

This is also true when `X` contains one sample.

## Evaluation

`Sequential`, `MLPClassifier`, and `MLPRegressor` expose:

```python
result = model.evaluate(X, y, metrics=("accuracy", "f1"))
# {"loss": 0.12, "metrics": {"accuracy": 0.95, "f1": 0.94}}
```

Metrics may be names, callables, or a mapping of names to callables. Multilabel
precision, recall, and F1 accept suffixes such as `precision_macro`,
`precision_micro`, and `precision_weighted`. Multiclass `roc_auc` uses a
one-vs-rest calculation.

## Checkpoints

`save_checkpoint()` and estimator `save()` write a versioned ZIP archive with a
JSON manifest and `.npy` arrays. The format does not load executable pickle
payloads. The manifest contains `format`, archive `schema_version`, a
`state_schema` version, and the library version. Unsupported archive or state
schema versions are rejected explicitly so a future loader can add migrations
without silently misreading model state. Callables are intentionally rejected
because they cannot be restored as a safe, portable object.

```python
model.save_checkpoint("model.checkpoint")
restored = Sequential.load_checkpoint("model.checkpoint")
```

## Backends

```python
mastermlx.set_backend("auto")    # compiled kernels when worthwhile
mastermlx.set_backend("numpy")   # deterministic fallback
mastermlx.set_backend("cython")  # require compiled Cython kernels
```

`backend_report()` reports requested, active, and available capabilities. In
`auto` mode, small recurrent workloads stay on NumPy to avoid the overhead of
element-wise compiled loops; explicit `cython` forces the compiled path.
