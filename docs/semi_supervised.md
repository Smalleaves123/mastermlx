# Semi-supervised learning

The package provides both transductive graph methods and an inductive
self-training classifier.

## Self-training

`SelfTrainingClassifier` wraps any classifier that implements `fit()` and
`predict_proba()`. Mark unknown targets with `-1` (or configure
`unlabeled_value`), then confident predictions are added to the next fitting
round:

```python
from mastermlx import LogisticRegression
from mastermlx.semi_supervised import SelfTrainingClassifier

model = SelfTrainingClassifier(
    LogisticRegression(lr=0.1, n_iter=500),
    threshold=0.85,
    max_iter=10,
).fit(X, y_with_unknowns)

predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
print(model.n_pseudo_labels_)
```

Use `criterion="k_best"` and `k_best=n` to add at most `n` high-confidence
samples per round. `labeled_iter_` records `0` for originally labeled
samples, the iteration that added pseudo-labels, and `-1` for samples that
remained unlabeled.

`LabelPropagation` and `LabelSpreading` remain useful when the complete
dataset is available at once and a transductive graph solution is preferred.
