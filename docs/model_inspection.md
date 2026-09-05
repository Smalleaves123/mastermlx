# Model inspection

`permutation_importance` measures the decrease in a fitted estimator's score
when one input feature is shuffled. It works with any estimator that follows
the `mastermlx` prediction and scoring contracts.

```python
from mastermlx import permutation_importance

result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="accuracy",
    n_repeats=10,
    random_state=0,
    feature_names=feature_names,
)
print(result["items"])
```

For an unbiased interpretation, use held-out or out-of-fold data rather than
the training rows. Positive importance means that permutation reduced model
quality; near-zero values indicate that the model did not rely on the feature
for that evaluation set.

`TabularExperiment.permutation_importance()` retains the result, and
`report(..., include_permutation_importance=True)` includes it in the unified
quality, drift, performance, calibration, and inspection report.

## Partial dependence and ICE

Use `partial_dependence` to inspect how a fitted model responds while one
feature varies across a grid:

```python
from mastermlx import partial_dependence

curves = partial_dependence(
    model,
    X_test,
    feature=0,
    grid_resolution=20,
    kind="both",       # average PDP plus individual ICE curves
    center=True,
)
```

Binary classifiers use the positive-class probability by default. Pass
`target=` for multiclass or multi-output estimators. Calling
`TabularExperiment.partial_dependence()` retains the result under the unified
report's `partial_dependence` section.
