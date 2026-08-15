# Classification Tutorial

[`compare_models.py`](compare_models.py) trains four binary classifiers on the
same reproducible dataset and draws their confusion matrices.

## Run

```bash
python -m pip install "mastermlx[viz]==0.1.15"
python examples/classification/compare_models.py
```

From the source checkout's development environment:

```bash
MPLBACKEND=Agg conda run -n CV python examples/classification/compare_models.py
```

The figure is saved as `examples/outputs/compare_models.png`.

## Interface

```python
from mastermlx.linear_models import LogisticRegression

classifier = LogisticRegression(lr=0.1, n_iter=200)
classifier.fit(X_train, y_train)
labels = classifier.predict(X_test)          # (n_samples,)
probabilities = classifier.predict_proba(X_test)  # when supported
accuracy = classifier.score(X_test, y_test)
```

`score` is accuracy for classifiers. Use a two-dimensional slice such as
`X_test[:1]` for one sample; `predict` still returns shape `(1,)`.

The demo also shows `SGDClassifier`, `RandomForestClassifier`, `SVC`, and
`mastermlx.viz.plot_cm`. See the
[`classification API index`](../API_REFERENCE.md#classification) for other
model families.
