import numpy as np

from mastermlx.neural_net import Dense, MLPRegressor, Sequential
from mastermlx.neural_net.metric_eval import evaluate_metrics
from mastermlx.utils.grad import accumulate_gradients, load_accumulated_gradients
from mastermlx.utils.metrics import roc_auc_score


def test_predict_keeps_batch_axis_and_evaluate_is_consistent():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    model = Sequential(
        [Dense(2, 4, random_state=0), Dense(4, 2, random_state=1)],
        task="classification",
        optimizer="adam",
        lr=0.03,
        n_iter=5,
        random_state=0,
    ).fit(X, y)

    assert model.predict(X[:1]).shape == (1,)
    assert model.predict_proba(X[:1]).shape == (1, 2)
    result = model.evaluate(X, y, metrics=("accuracy", lambda yt, yp: np.mean(yt == yp)))
    assert set(result) == {"loss", "metrics"}
    assert set(result["metrics"]) == {"accuracy", "<lambda>"}


def test_multiclass_auc_and_multilabel_averages():
    y = np.array([0, 1, 2, 0, 1, 2])
    score = np.array([
        [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7],
    ])
    assert np.isclose(roc_auc_score(y, score, labels=np.array([0, 1, 2])), 1.0)
    values = evaluate_metrics("classification", ("roc_auc",), y, np.log(score), classes=np.array([0, 1, 2]))
    assert np.isclose(values["roc_auc"], 1.0)

    target = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    output = np.array([[2, -1], [1, 2], [-1, 1], [-2, -1]], dtype=float)
    values = evaluate_metrics(
        "multilabel",
        ("precision_macro", "precision_micro", "precision_weighted"),
        target,
        output,
    )
    assert all(np.isclose(value, 1.0) for value in values.values())


def test_mlp_regressor_supports_multioutput_targets():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.column_stack([X[:, 0] + X[:, 1], X[:, 0] - X[:, 1]])
    model = MLPRegressor(hidden_layer_sizes=(), n_iter=3, lr=0.001, random_state=0).fit(X, y)

    assert model.predict(X[:1]).shape == (1, 2)
    result = model.evaluate(X, y, metrics={"max_error": lambda yt, yp: np.max(np.abs(yt - yp))})
    assert result["metrics"]["max_error"] >= 0.0


def test_accumulation_uses_sample_weight():
    layer = Dense(2, 1, random_state=0)
    layer.dW_ = np.ones_like(layer.W_)
    layer.db_ = np.ones_like(layer.b_)
    store = {}
    accumulate_gradients([layer], store, weight=3)
    layer.dW_ = np.full_like(layer.W_, 3.0)
    layer.db_ = np.full_like(layer.b_, 3.0)
    accumulate_gradients([layer], store, weight=1)
    load_accumulated_gradients([layer], store, weight=4)

    assert np.allclose(layer.dW_, 1.5)
    assert np.allclose(layer.db_, 1.5)
