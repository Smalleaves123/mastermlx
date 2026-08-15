# Probabilistic Models Tutorial

[`probabilistic_models.py`](probabilistic_models.py) demonstrates two common
interfaces: Gaussian-process regression with predictive uncertainty and linear
discriminant analysis for classification.

## Run

```bash
python -m pip install "mastermlx==0.1.15"
python examples/probabilistic/probabilistic_models.py
```

## Regression uncertainty

```python
from mastermlx.probabilistic import GaussianProcessRegressor

model = GaussianProcessRegressor(length_scale=1.0, alpha=1e-5).fit(X, y)
mean, std = model.predict(X_query, return_std=True)
samples = model.sample_posterior_functions(X_query, n_samples=10, random_state=0)
```

`alpha` is observation/noise regularization. Predictive standard deviation is
typically larger away from training observations.

## Classification

```python
from mastermlx.probabilistic import DiscriminantLDA

classifier = DiscriminantLDA().fit(X_train, y_train)
labels = classifier.predict(X_test)
accuracy = classifier.score(X_test, y_test)
```

The explicit name avoids confusion with `mastermlx.nlp.NLP_LDA`, the unrelated
topic model. See the
[`probabilistic API index`](../API_REFERENCE.md#probabilistic-models) for naive
Bayes, QDA, Bayesian regression, HMM, KDE, and distributions.
