import numpy as np

from mastermlx.neural_net import (
    AttentionPooling1D,
    BatchNorm,
    BinaryCrossEntropyLoss,
    Dense,
    GRU,
    LayerNorm,
    LSTM,
    MSELoss,
    MultiHeadAttention,
    OptimizerConfig,
    ReLU,
    Sequential,
    SimpleRNN,
)
from mastermlx.utils.grad import accumulate_gradients, clip_gradients


def _input_grad(layer, X, seed=0, eps=1e-5):
    rng = np.random.default_rng(seed)
    out = layer.forward(X)
    grad = rng.normal(size=out.shape)
    analytic = layer.backward(grad)
    numeric = np.zeros_like(X)

    for index in np.ndindex(X.shape):
        original = X[index]
        X[index] = original + eps
        plus = np.sum(layer.forward(X) * grad)
        X[index] = original - eps
        minus = np.sum(layer.forward(X) * grad)
        X[index] = original
        numeric[index] = (plus - minus) / (2.0 * eps)

    assert np.allclose(analytic, numeric, atol=3e-5, rtol=3e-4)


def _param_grad(layer, X, name, seed=0, eps=1e-5):
    rng = np.random.default_rng(seed)
    out = layer.forward(X)
    grad = rng.normal(size=out.shape)
    layer.backward(grad)
    analytic = np.asarray(getattr(layer, "d" + name), dtype=float)
    original = np.asarray(getattr(layer, name), dtype=float).copy()
    numeric = np.zeros_like(original)

    for index in np.ndindex(original.shape):
        plus = original.copy()
        minus = original.copy()
        plus[index] += eps
        minus[index] -= eps
        setattr(layer, name, plus)
        value_plus = np.sum(layer.forward(X) * grad)
        setattr(layer, name, minus)
        value_minus = np.sum(layer.forward(X) * grad)
        numeric[index] = (value_plus - value_minus) / (2.0 * eps)

    setattr(layer, name, original)
    assert np.allclose(analytic, numeric, atol=3e-5, rtol=3e-4)


def test_rnn_xgrad():
    X = np.random.default_rng(1).normal(size=(1, 3, 2))
    for layer in (
        SimpleRNN(2, return_sequences=True, random_state=2),
        LSTM(2, return_sequences=True, random_state=2),
        GRU(2, return_sequences=True, random_state=2),
    ):
        _input_grad(layer, X.copy())


def test_norm_attn_xgrad():
    rng = np.random.default_rng(3)
    _input_grad(BatchNorm(3), rng.normal(size=(4, 3)))
    _input_grad(LayerNorm(3), rng.normal(size=(2, 3)))
    _input_grad(AttentionPooling1D(hidden_size=3, random_state=4), rng.normal(size=(1, 3, 2)))
    _input_grad(MultiHeadAttention(n_features=2, n_heads=1, random_state=5), rng.normal(size=(1, 3, 2)))


def test_layer_param_grads():
    X = np.random.default_rng(13).normal(size=(1, 3, 2))
    for layer, names in (
        (SimpleRNN(2, return_sequences=True, random_state=14), ("W_xh_", "W_hh_", "b_")),
        (LSTM(2, return_sequences=True, random_state=14), ("W_", "U_", "b_")),
        (GRU(2, return_sequences=True, random_state=14), ("W_zr_", "W_h_", "U_zr_", "U_h_", "b_zr_", "b_h_")),
        (BatchNorm(2), ("gamma_", "beta_")),
        (LayerNorm(2), ("gamma_", "beta_")),
        (AttentionPooling1D(hidden_size=2, random_state=14), ("W_", "b_", "u_", "c_")),
        (MultiHeadAttention(n_features=2, n_heads=1, random_state=14), ("W_q_", "W_k_", "W_v_", "W_o_", "b_q_", "b_k_", "b_v_", "b_o_")),
    ):
        data = X
        if isinstance(layer, (BatchNorm, LayerNorm)):
            data = np.random.default_rng(15).normal(size=(4, 2))
        for name in names:
            _param_grad(layer, data.copy(), name)


def test_loss_mean_grads():
    cases = [
        (
            MSELoss(),
            np.array([[0.2, -0.3], [0.7, 0.4]]),
            np.array([[0.4, -0.1], [0.2, 0.9]]),
        ),
        (
            BinaryCrossEntropyLoss(),
            np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            np.array([[0.2, -0.4, 1.1], [-0.3, 0.5, 0.7]]),
        ),
    ]
    eps = 1e-6
    for loss, y_true, y_pred in cases:
        analytic = loss.grad(y_true, y_pred)
        numeric = np.zeros_like(y_pred)
        for index in np.ndindex(y_pred.shape):
            plus = y_pred.copy()
            minus = y_pred.copy()
            plus[index] += eps
            minus[index] -= eps
            numeric[index] = (loss(y_true, plus) - loss(y_true, minus)) / (2.0 * eps)
        assert np.allclose(analytic, numeric, atol=3e-6, rtol=3e-5)


def test_grad_tools_cover_layers():
    X = np.random.default_rng(6).normal(size=(1, 3, 2))
    layers = [
        SimpleRNN(2, random_state=7),
        GRU(2, random_state=8),
        MultiHeadAttention(2, random_state=9),
    ]
    for layer in layers:
        out = layer.forward(X)
        layer.backward(np.ones_like(out))

    store = {}
    for layer in layers:
        accumulate_gradients([layer], store)
        for name, value in vars(layer).items():
            if name.startswith("d") and name.endswith("_") and value is not None:
                assert (0, name) in store
                break

    total_norm, scale = clip_gradients(layers, max_norm=0.5)
    assert total_norm > 0.5
    assert scale < 1.0


def test_bn_adam_stable():
    X = np.array([
        [-2.0, -1.5], [-1.5, -1.0], [-1.0, -2.0], [-0.5, -1.2],
        [1.0, 1.5], [1.5, 1.0], [2.0, 1.2], [0.8, 2.0],
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = Sequential(
        [Dense(2, 6, random_state=10), BatchNorm(6), ReLU(), Dense(6, 2, random_state=11)],
        optimizer="adam",
        optimizer_config=OptimizerConfig(name="adam", lr=0.01),
        n_iter=80,
        batch_size=8,
        tol=0.0,
        shuffle=False,
        random_state=12,
        task="classification",
    )
    model.fit(X, y)

    losses = np.asarray(model.loss_, dtype=float)
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0]
