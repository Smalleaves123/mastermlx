# Neural Network Tutorial

[`mlp_spirals.py`](mlp_spirals.py) trains an `MLPClassifier` on a nonlinear
two-class dataset and plots the learned decision regions.

## Run

```bash
python -m pip install "mastermlx[viz]==0.1.15"
MPLBACKEND=Agg python examples/neural_networks/mlp_spirals.py
```

The figure is saved as `examples/outputs/mlp_spirals.png`.

## High-level estimator interface

```python
from mastermlx.neural_net import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(32, 32),
    n_iter=200,
    random_state=0,
)
model.fit(X_train, y_train)
labels = model.predict(X_test)
accuracy = model.score(X_test, y_test)
```

Use `Sequential` when you need explicit layers, optimizers, callbacks, or
schedulers. The public layer set includes dense, convolutional, recurrent,
normalization, embedding, and attention components.

See the [`neural API contract`](../../docs/neural_api.md) for tensor shapes,
training, evaluation results, optimizers, persistence, and backends. The
[`0.1.15 API guide`](../API_REFERENCE.md#neural-networks) provides the concise
interface index.
